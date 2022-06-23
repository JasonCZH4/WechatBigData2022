from functools import partial
from xbert import BertConfig, BertModel_xbert
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel
from layer import NeXtVLAD, SENet, ConcatDenseSE, ECA_Layer, ConcatDenseECA, MultiModalResidual
from category_id_map import CATEGORY_ID_LIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()
        bert_config = BertConfig.from_pretrained('chinese-roberta-wwm-ext/config.json')

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(768, 768)
        self.enhance = ECA_Layer(768)
        self.projection = nn.Linear(768, 768)
        self.ln = nn.LayerNorm([384, 768])
        self.ln1d = nn.LayerNorm(768)
        # self.video_fc = torch.nn.Linear(768, bert_config.hidden_size)
        self.video_embeddings = BertEmbeddings(bert_config)
        self.video_encoder = BertEncoder(bert_config)
        self.text_encoder = BertModel_xbert.from_pretrained(args.bert_dir, config=bert_config, add_pooling_layer=False)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.cls_ln = nn.LayerNorm(7680)
        self.cls_head = nn.Linear(7680, 200)

    def forward(self, inputs, merge=False, train=True, inference=False):
        vlad_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])

        output = self.text_encoder(inputs['title_input'],
                                   attention_mask=inputs['title_mask'],
                                   token_type_ids=inputs['token_type_ids'],
                                   encoder_hidden_states=inputs['frame_input'],
                                   encoder_attention_mask=inputs['frame_mask'],
                                   return_dict=True
                                   )
        last_hidden_state = output.last_hidden_state
        mean_last_hidden = (last_hidden_state * inputs['title_mask'].unsqueeze(-1)).sum(1) \
                           / inputs['title_mask'].sum(1).unsqueeze(-1)
        mean_last_hidden = mean_last_hidden.float()
        ln_mean_last_hidden = self.ln1d(mean_last_hidden)
        max_last_hidden = last_hidden_state + (1 - inputs['title_mask']).unsqueeze(-1) * (-1e10)
        max_last_hidden = max_last_hidden.max(1)[0].float()
        ln_max_last_hidden = self.ln1d(max_last_hidden)
        cls_info = last_hidden_state[:, 0, :]
        pooler_output = self.tanh(self.dense(cls_info))
        pooler_attention = self.enhance(pooler_output.unsqueeze(dim=1))
        mean_output = torch.mean((pooler_attention*last_hidden_state), 1)
        ln_output = self.ln1d(mean_output)
        fusion_embedding = self.cls_ln(torch.cat(
            (ln_output, pooler_output, ln_mean_last_hidden, ln_max_last_hidden, cls_info, vlad_embedding), dim=1))
        prediction = self.cls_head(fusion_embedding)
        if inference:
            return torch.argmax(prediction, dim=1)
        elif merge:
            return prediction
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

