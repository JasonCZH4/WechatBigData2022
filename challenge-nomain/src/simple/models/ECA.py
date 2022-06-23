import torch
import torch.nn as nn


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

