import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from models.mae import Transformer
from models.utils import get_2d_sincos_pos_embed


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# # ITM head, two cls concate to predict something
class ITMHead(nn.Module):
    def __init__(self, hidden_size,out_size=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# MLM 经过BertPredictionHeadTransform层 + 线性层  hidden_size -> vocab_size
'''
BertPredictionHeadTransform:为
nn.Linear(config.hidden_size, config.hidden_size)
ACT2FN[config.hidden_act]
BertLayerNorm
'''
class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

# fingerprint predict head,一个全连接层 cls_feats
class FPPHead(nn.Module):
    def __init__(self, hidden_size,out_size=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# MoleculeClassify head, two cls concate to predict something
class MoleculeClassify(nn.Module):
    def __init__(self, hidden_size,drop_rate,out_size=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# MoleculeRegress head, two cls concate to predict something
class MoleculeRegress(nn.Module):
    def __init__(self, hidden_size,drop_rate):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.fc(x)
        return x