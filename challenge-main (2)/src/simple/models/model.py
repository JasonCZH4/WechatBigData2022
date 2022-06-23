# %%writefile qqmodel/qq_uni_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.simple.config.category_id_map import CATEGORY_ID_LIST
import sys

sys.path.append("../../..")
from transformers.models.bert.modeling_bert import BertConfig, BertPooler, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder


class QQModel3(nn.Module):
    def __init__(self, cfg, model_path):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'config/config.json')
        # uni_bert_cfg.num_hidden_layers = 1

        self.Classifier = torch.nn.Linear(cfg.fc_size*4, len(CATEGORY_ID_LIST))
        self.roberta = UniBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)

    def forward(self, inputs, inference=False):


        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']

        # concat features
        features = self.roberta(video_feature, video_mask, text_input_ids, text_mask)
        prediction = self.Classifier(features)  # (16,200)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        encoder_outputs = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        return encoder_outputs



class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()
        self.enhanceSE = SENet(channels=config.hidden_size, ratio=config.se_ratio)
        self.cls_ln = nn.LayerNorm(config.hidden_size * 4)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.ln = nn.LayerNorm(config.hidden_size)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):
        text_emb = self.embeddings(input_ids=text_input_ids)

        video_emb = torch.sigmoid(self.video_embeddings(inputs_embeds=video_feature))
        fusion_emb = torch.cat((text_emb, video_emb), 1)
        # vlad_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        # fusion_emb = self.dropout(fusion_emb)
        # fusion_emb = self.enhanceSE(fusion_emb)
        fusion_mask = torch.cat((text_mask, video_mask), dim=1)  # (bs, seq+32)
        fusion_mask_attention = fusion_mask[:, None, None, :]   # (16,1,1,seq+32)
        fusion_mask_attention = (1.0 - fusion_mask_attention) * -10000.0
        last_hidden_state = self.encoder(fusion_emb, attention_mask=fusion_mask_attention)['last_hidden_state']

        last4_hidden = self.encoder(fusion_emb, attention_mask=fusion_mask_attention, output_hidden_states=True)['hidden_states'][-4]

        # last4 meanpooling
        embed_mean_fusion = (last4_hidden * fusion_mask.unsqueeze(-1)).sum(1) / fusion_mask.sum(1).unsqueeze(-1)
        embed_mean_fusion = embed_mean_fusion.float()  # (bs,768))
        embed_mean_fusion = self.ln(embed_mean_fusion)

        # pooler_outputs,attention,lastmeanpooling
        pooler_outputs = self.pooler(last_hidden_state)  # (bs, 768)
        attention = self.enhanceSE(pooler_outputs)  # (bs,1,768)
        embed_mean_hidden = (last_hidden_state*fusion_mask.unsqueeze(-1)).sum(1)/fusion_mask.sum(1).unsqueeze(-1)
        embed_mean_hidden = embed_mean_hidden.float()  # (bs,768)
        embed_mean_hidden = self.ln(embed_mean_hidden)
        # embed_max_hidden = last_hidden_state+(1-fusion_mask).unsqueeze(-1)*(-1e10)
        # embed_max_hidden = embed_max_hidden.max(1)[0].float()

        # cat
        embed_mean = self.cls_ln(torch.cat((embed_mean_fusion, embed_mean_hidden, pooler_outputs, attention), dim=1))  # (bs, 3072)
        return embed_mean



def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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






