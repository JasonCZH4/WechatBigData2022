import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=768, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups  # 96

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])    # bs, 96 * 64
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class ECA_Layer(nn.Module):
    """
    channel: Number of channels of the input feature map
    k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_Layer, self).__init__()
        self.channel = channel
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, length, c = x.size()     # 64, 82, 768 or 1024
        if length != 1:
            y = self.relu(torch.mean(x, 1, keepdim=True))  # 64, 1, 768 or 1024
        else:
            y = self.relu(x)
        y2 = self.conv(y)
        y3 = self.sigmoid(y2)
        return x * y3.expand_as(x)


class ConcatDenseECA(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(6144)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(6144, 3072)
        self.enhance = ECA_Layer(channel=3072)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)   # bs, 4608
        embeddings = self.ln(embeddings)    # bs, 4608
        embeddings = self.fusion_dropout(embeddings)    # bs, 4608
        embeddings = self.fc(embeddings)    # bs, 1024
        embedding = self.enhance(embeddings.unsqueeze(dim=1)).squeeze()  # bs, 1024

        return embedding


class MultiModalResidual(nn.Module):
    def __init__(self, word_embedding, video_embedding):
        super().__init__()
        self.word_fc = nn.Linear(2304, 768)
        self.video_fc1 = nn.Linear(2304, 1024)
        self.video_fc2 = nn.Linear(1024, 768)
        self.tanh = nn.Tanh()

    def forward(self, word_inputs, video_inputs):
        word_input = torch.cat(word_inputs, dim=1)  # bs, 2304
        residual = self.word_fc(word_input)    # bs, 768
        word_input = self.tanh(self.word_fc(word_input))    # bs, 768
        video_input = self.tanh(self.video_fc2(self.tanh(self.video_fc1(video_inputs))))   # bs, 768
        dot_result = word_input * video_input
        output = residual + dot_result  # bs, 768

        return output

