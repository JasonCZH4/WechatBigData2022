import torch

import torch.nn as nn


class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def get_attn(self, reps, mask=None):
        reps = torch.unsqueeze(reps, 1)
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = mask * attn_scores
        attn_weight = attn_scores.unsqueeze(2)
        attn_output = torch.sum(reps * attn_weight, dim=1)

        return attn_output

    def forward(self, reps, mask=None):

        attn_output = self.get_attn(reps, mask)
        return attn_output
