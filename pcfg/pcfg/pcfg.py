import os, re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from collections import OrderedDict
from fvcore.common.registry import Registry
from omegaconf.listconfig import ListConfig

from .base import PCFG, ResLayer
from ..module import PretrainedEncoder, PartiallyFixedEmbedding

class NaivePCFG(PCFG):
    def __init__(self, cfg, NT=0, T=0, vocab=None, skip_init=False, **kwargs):
        super(NaivePCFG, self).__init__(cfg, NT=NT, T=T, vocab=vocab, **kwargs)
        if skip_init: return #
        self.cfg = cfg
        h_dim = cfg.h_dim
        w_dim = cfg.w_dim
        z_dim = cfg.z_dim
        s_dim = cfg.s_dim

        assert z_dim >= 0

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if cfg.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if cfg.share_root else s_dim + z_dim
        root_modules = ( 
            nn.Linear(root_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, NT),
        ) 
        self.root_mlp = nn.Sequential(*root_modules)

        self._build_z_encoder(vocab)

        term_dim = s_dim if cfg.share_term else s_dim + z_dim
        term_modules = ( 
            nn.Linear(term_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, len(vocab)),
        ) 
        self.term_mlp = nn.Sequential(*term_modules)
        self._count_rnd_consumed()
        self._initialize()

        if cfg.tied_terms and isinstance(self.enc_emb, PartiallyFixedEmbedding):
            self.term_mlp[-1] = self.enc_emb

    def forward(
        self, x, lengths, *args, sub_words=None, token_indice=None, 
        use_mean=False, max_pooling=True, enforce_sorted=False, **kwargs
    ):
        """ x, lengths: words; sub_words, token_indice: sub-word 
        """
        b, n = x.shape[:2]
        if self.z_dim > 0:
            item, fn = x, self.enc # default z encoder
            if isinstance(self.enc_emb, PretrainedEncoder):
                item = (x, lengths, sub_words, token_indice)
                fn = self.enc_with_loss_mlm if self.cfg.multi_view else self.enc_semantic
            elif isinstance(self.enc_emb, PartiallyFixedEmbedding):
                fn = self.enc_lexicons
            mean, lvar, extra = fn(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
            extra = (mean, lvar, extra)
        else:
            z = torch.zeros(b, 1).cuda()
            kl = extra = None
        self.z = z

        def roots(pcfg=False):
            root_emb = self.root_emb.expand(b, self.s_dim)
            if not pcfg and self.z_dim > 0 and not self.cfg.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            mlp = self.root_mlp
            root_prob = F.log_softmax(mlp(root_emb), -1)
            return root_prob
        
        def terms(pcfg=False):
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if not pcfg and self.z_dim > 0 and not self.cfg.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            mlp = self.term_mlp
            term_prob = F.log_softmax(mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules(pcfg=False):
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            if not pcfg and self.z_dim > 0 and not self.cfg.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            mlp = self.rule_mlp
            rule_prob = F.log_softmax(mlp(nonterm_emb), -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        
        extra = tuple()
        return (terms_ll, rules_ll, roots_ll), kl, extra 

