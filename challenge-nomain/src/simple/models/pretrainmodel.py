# %%writefile qqmodel/qq_uni_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.simple.config.category_id_map import CATEGORY_ID_LIST
import sys

sys.path.append("../../..")
from src.simple.models.masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead, BertPooler
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder


class QQModel4(nn.Module):
    def __init__(self, cfg, model_path):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'config/config.json')
        # uni_bert_cfg.num_hidden_layers = 1

        self.newfc_hidden = torch.nn.Linear(uni_bert_cfg.hidden_size, cfg.bert_hidden_size)
        self.classifier = torch.nn.Linear(uni_bert_cfg.hidden_size, len(CATEGORY_ID_LIST))

        # mlm
        self.lm = MaskLM(tokenizer_path=model_path)
        self.num_class = len(CATEGORY_ID_LIST)
        self.vocab_size = uni_bert_cfg.vocab_size

        # mfm
        self.vm = MaskVideo()
        self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg)

        # itm
        self.sv = ShuffleVideo()
        self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1)

        self.roberta = UniBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)

    def forward(self, inputs, inference=False, task=None):
        loss, pred = 0, None

        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']


        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        # mlm
        input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
        text_input_ids = input_ids.to(text_input_ids.device)
        lm_label = lm_label[:, 1:].to(text_input_ids.device)  # [SEP] 卡 MASK 大师 [SEP]
        return_mlm = True

        # mfm
        vm_input = video_feature
        input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
        video_feature = input_feature.to(video_feature.device)
        video_label = video_label.to(video_feature.device)

        # itm
        input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
        video_feature = input_feature.to(video_feature.device)
        video_text_match_label = video_text_match_label.to(video_feature.device)

        # concat features
        features, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask,
                                                                return_mlm=return_mlm)

        # compute pretrain task loss

        # mlm
        pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
        masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
        loss += masked_lm_loss / 3.75

        # mfm
        vm_output = self.roberta_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
        masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input,
                                                 video_mask, video_label, normalize=False)
        loss += masked_vm_loss / 9

        # itm
        pred = self.newfc_itm(features[:, 0, :])
        itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
        loss += itm_loss / 3
        return loss, masked_lm_loss, masked_vm_loss, itm_loss

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        # criterion = LabelSmoothingCrossEntropy()
        # loss = criterion(prediction, label)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    # calc mfm loss
    def calculate_mfm_loss(self, video_feature_output, video_feature_input,
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss



def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


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


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        encoder_outputs = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]:, :]
        else:
            return \
                encoder_outputs, None


class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.enhance = SENet(channels=config.hidden_size, ratio=config.se_ratio)
        self.embeddings = BertEmbeddings(config)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):
        text_emb = self.embeddings(input_ids=text_input_ids)

        video_emb = torch.sigmoid(self.video_embeddings(inputs_embeds=video_feature))
        fusion_emb = torch.cat((text_emb, video_emb), 1)

        fusion_mask = torch.cat((text_mask, video_mask), dim=1)
        fusion_mask_attention = fusion_mask[:, None, None, :]
        fusion_mask_attention = (1.0 - fusion_mask_attention) * -10000.0

        # pooler_outputs,attention,lastmeanpooling
        last_hidden_state =self.encoder(fusion_emb, attention_mask=fusion_mask_attention)['last_hidden_state']  # (bs,seq+32,768)

        return last_hidden_state
