from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...module import (
    LayerNorm, PretrainedEncoder, PartiallyFixedEmbedding,
    layernorm_linear, linear_relu_linear
)

TEXT_HEADS_REGISTRY = Registry("TEXT_HEADS")
TEXT_HEADS_REGISTRY.__doc__ = """
Registry for text encoders.
"""

def build_text_head(cfg, **kwargs):
    return TEXT_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

@TEXT_HEADS_REGISTRY.register()
class DummyHead(torch.nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.num_state = -1
        pass
    def from_pretrained(self, state_dict, cfg, *args, **kwargs):
        pass
    def copy_state_dict(self, state_dict):
        return {}, {}
    def replace_modules(self, **kwargs):
        return []
    def forward(self, x, *args, **kwargs):
        z = x # do nothing
        if kwargs.get("normalized", False):
            z = F.normalize(z, dim=-1) #z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} linear --{kwargs.get('normalized', False)}")
        return z

@TEXT_HEADS_REGISTRY.register()
class LinearHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.num_state = -1
        sizes = [cfg.input_dim] + list(cfg.layers) + [cfg.embed_dim]
        layers = layernorm_linear(
            sizes, cfg.layer_norm, cfg.ibias, cfg.bias
        )
        self.encoder = nn.Sequential(*layers)
        self._output_dim = cfg.embed_dim
        self._num_rnd_consumed = 0
        self._count_rnd_consumed()
        self._initialize()

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def num_rnd_consumed(self):
        return self._num_rnd_consumed

    def _initialize(self): 
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
                self._num_rnd_consumed += torch.numel(p)

    def _count_rnd_consumed(self):
        for layer in self.encoder:
            if not isinstance(layer, nn.Linear):
                continue
            for p in layer.parameters():
                self._num_rnd_consumed += torch.numel(p)

    def forward(self, text, *args, **kwargs):
        z = self.encoder(text)
        if kwargs.get("normalized", False):
            z = F.normalize(z, dim=-1) #z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} linear --{kwargs.get('normalized', False)}")
        return z 

@TEXT_HEADS_REGISTRY.register()
class PCFGFusionHead(LinearHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        sizes = [cfg.input_dim] + list(cfg.layers) + [cfg.embed_dim]
        layers = linear_relu_linear(
            sizes, cfg.layer_norm, cfg.ibias, cfg.bias
        )
        self.encoder = nn.Sequential(*layers)
        self._initialize()

    def forward(self, text, *args, **kwargs):
        assert "pcfg_head" in kwargs, f"PCFG head is not found."
        pcfg_head = kwargs["pcfg_head"]
        nonterm_emb = pcfg_head.nonterm_emb
        nonterm_emb = nonterm_emb[:pcfg_head.NT]

        text = text @ nonterm_emb

        z = self.encoder(text)
        if kwargs.get("normalized", False):
            z = F.normalize(z, dim=-1) #z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} linear --{kwargs.get('normalized', False)}")
        return z

@TEXT_HEADS_REGISTRY.register()
class SRNNTextEncoder(torch.nn.Module):
    def __init__(self, cfg, vocab=None, enc_emb=None, **kwargs):
        super().__init__()
        self.num_state = cfg.num_state
        self.span_pooler = cfg.span_pooler
        self.enc_rnn = torch.nn.LSTM(
            cfg.w_dim, cfg.h_dim, bidirectional=True, num_layers=1, batch_first=True
        )
        self.enc_out = nn.Linear(cfg.h_dim * 2, cfg.embed_dim * self.num_state, bias=cfg.bias)
        self.enc_emb = nn.Embedding(len(vocab), cfg.w_dim)
        self._output_dim = cfg.embed_dim
        self._num_rnd_consumed = 0
        self._count_rnd_consumed()
        self._initialize()
        # word emb sharing 
        if enc_emb is not None:
            self.enc_emb = enc_emb

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def num_rnd_consumed(self):
        return self._num_rnd_consumed

    def _initialize(self): 
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
                self._num_rnd_consumed += torch.numel(p)

    def _count_rnd_consumed(self):
        for k, p in self.named_parameters():
            self._num_rnd_consumed += torch.numel(p)

    def mean_span(self, word_emb):
        device = word_emb.device
        B, N, H = word_emb.size()
        spans = torch.zeros(
            B, int(N * (N - 1) / 2), self.num_state, self.output_dim, device=device
        )
        beg_idx = 0 
        for k in range(1, N):
            inc = torch.arange(N - k, device=device).view(N - k, 1)
            idx = torch.arange(k + 1, device=device).view(1, k + 1).repeat(N - k, 1)
            idx = (idx + inc).view(-1)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, H) 

            span = torch.gather(word_emb, 1, idx)
            span = span.view(B, N - k, k + 1, H)
            span = span.view(-1, k + 1, H) 
            span = self.enc_rnn(span)[0]
            span = self.enc_out(span)
            span = span.view(B, N - k, k + 1, self.num_state, self.output_dim)
            span = span.mean(2)
            end_idx = beg_idx + N - k 
            spans[:, beg_idx : end_idx] = span 
            beg_idx = end_idx
        return spans

    def forward(self, x, lengths, *args, token_indice=None, sub_words=None, **kwargs):
        if isinstance(self.enc_emb, PretrainedEncoder):
            x = (x, lengths, sub_words, token_indice)
        word_emb = self.enc_emb(x)
        if self.span_pooler == "mean":
            z = self.mean_span(word_emb)
        else:
            raise ValueError(f"unsupported span pooler {self.span_pooler}")

        if kwargs.get("normalized", False):
            z = F.normalize(z, dim=-1) #z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} srnn --{kwargs.get('normalized', False)}")
        return z 
