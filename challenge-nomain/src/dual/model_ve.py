from functools import partial
# from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
import torch
from torch import nn
import torch.nn.functional as F
from soft_attention import SoftAttention


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()

        # bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config = BertConfig.from_pretrained('../input/pretrain-model/chinese-macbert-base/config.json')
        self.video_fc = torch.nn.Linear(768, bert_config.hidden_size)
        self.video_embeddings = BertEmbeddings(bert_config)
        self.video_encoder = BertEncoder(bert_config)
        self.text_encoder = BertModel.from_pretrained(args.bert_dir, config=bert_config, add_pooling_layer=False)

        self.cls_head = nn.Sequential(
            nn.Linear(3840, 200)
        )

        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.soft_attention = SoftAttention(3840)

    def forward(self, inputs, alpha=0, train=True, inference=False):

        video_feature = self.video_fc(inputs['frame_input'])
        video_emb = self.video_embeddings(inputs_embeds=video_feature)
        att_mask = inputs['frame_mask'][:, None, None, :]
        att_mask = (1.0 - att_mask) * -10000.0
        encoder_outputs1 = self.video_encoder(video_emb, att_mask, output_hidden_states=True)
        # image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(encoder_outputs1['hidden_states'][-1].size()[:-1],dtype=torch.long).to(device)


        output = self.text_encoder(inputs['text_input'],
                                   attention_mask=inputs['text_mask'],
                                   encoder_hidden_states=encoder_outputs1['hidden_states'][-1],
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   output_hidden_states=True
                                   )
        output = torch.cat([output['hidden_states'][-1], output['hidden_states'][-2],
                                      output['hidden_states'][-3], output['hidden_states'][-4]], 2)


        mask = inputs['text_mask']

        pooling_output = torch.einsum("bsh,bs,b->bh", output, mask.float(),
                                      1 / mask.float().sum(dim=1) + 1e-9)

        vlad_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])

        vision_embedding = self.soft_attention(torch.cat([pooling_output, vlad_embedding], 1))

        prediction = self.cls_head(vision_embedding)
        if train:
            return self.cal_loss(prediction, inputs['label'])
        elif inference:
            return torch.argmax(prediction, dim=1)
        else:
            return prediction

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)




class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

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
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
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
