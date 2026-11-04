"""Authorized dataset-specific feature encoders and shared classifiers.

Numeric feature tokens are pooled into eight documented feature groups. Recurrent
models operate on group sequences, NOT temporal network sequences. This is an
adapted tabular architecture experiment, not a temporal IDS validation.
"""
import torch
import re
from torch import nn
from torch.nn import functional as F


GROUPS = ['ports', 'packet_counts', 'size_statistics', 'timing_rates',
          'flags', 'protocol_attributes', 'headers_topology', 'other']


def group_for(feature):
    name = feature.lower()
    if 'port' in name:
        return 0
    if 'flag' in name or 'checksum' in name or name == 'is_with_payload' or re.search(r'(^|[_. -])(syn|ack|fin|rst|psh|ece|cwr)([_. -]|$)', name):
        return 4
    if name == 'iat' or any(s in name for s in ['rate','interval','time-delta','time_delta','duration']):
        return 3
    if any(s in name for s in ['header','ttl','time_to_live','mss','window','fragment','ips_','macs_']):
        return 6
    if any(s in name for s in ['bytes','length','size','len','min','max','avg','std','variance','tot sum']):
        return 2
    if any(s in name for s in ['pkts','packts','packets']) or name == 'number' or name == 'total_packet':
        return 1
    if 'protocol' in name or name in {'http','https','dns','telnet','smtp','ssh','irc','tcp','udp','dhcp','arp','icmp','igmp','ipv','llc'} or name.startswith(('mqtt.','dns.','mbtcp.','arp.','http.')):
        return 5
    return 7


class FeatureEncoder(nn.Module):
    def __init__(self, feature_names, width=8):
        super().__init__()
        features = len(feature_names)
        self.weight = nn.Parameter(torch.randn(features, width) * .02)
        self.bias = nn.Parameter(torch.randn(features, width) * .02)
        pooling = torch.zeros(8, features)
        for j, name in enumerate(feature_names):
            pooling[group_for(name), j] = 1
        pooling /= pooling.sum(dim=1, keepdim=True).clamp_min(1)
        self.register_buffer('pooling', pooling)

    def forward(self, x):
        tokens = torch.tanh(x.unsqueeze(-1) * self.weight + self.bias)
        return torch.einsum('gf,bfe->bge', self.pooling, tokens)


class AdaptedModel(nn.Module):
    def __init__(self, kind, feature_lists, dropout=.3):
        super().__init__()
        self.kind = kind
        self.max_features = 8
        self.encoders = nn.ModuleList(FeatureEncoder(names) for names in feature_lists)
        if kind == 'MLP':
            self.core = nn.Sequential(nn.Flatten(), nn.Linear(self.max_features*8, 128),
                                      nn.ReLU(), nn.Dropout(dropout))
            width = 128
        elif kind == 'CNN1D':
            self.core = nn.Sequential(nn.Conv1d(8, 128, 3, padding=1), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Conv1d(128, 64, 3, padding=1),
                                      nn.ReLU(), nn.Dropout(dropout),
                                      nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten())
            width = 32
        elif kind in {'LSTM', 'GRU', 'BiLSTM'}:
            cls = nn.GRU if kind == 'GRU' else nn.LSTM
            self.core = cls(8, 128, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=kind == 'BiLSTM')
            width = 256 if kind == 'BiLSTM' else 128
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def forward(self, x, domain):
        tokens = self.encoders[domain](x)
        if self.kind == 'MLP':
            representation = self.core(F.pad(tokens, (0, 0, 0, self.max_features-tokens.shape[1])))
        elif self.kind == 'CNN1D':
            representation = self.core(tokens.transpose(1, 2).contiguous())
        else:
            _, state = self.core(tokens)
            hidden = state[0] if isinstance(state, tuple) else state
            representation = (torch.cat([hidden[-2], hidden[-1]], dim=1)
                              if self.kind == 'BiLSTM' else hidden[-1])
        return self.head(representation).squeeze(-1)
