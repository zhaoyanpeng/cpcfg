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

from .base import ResLayer
from .vqpcfg import VQPCFG
from ..module import soft_topk
from ..module import VectorQuantizer, VectorQuantizerEMA 
from ..module import PretrainedEncoder, PartiallyFixedEmbedding

class NystromPCFG(VQPCFG):
    def __init__(self, cfg, NT=0, T=0, vocab=None, **kwargs): 
        super(NystromPCFG, self).__init__(
            cfg, NT=NT, T=T, vocab=vocab, skip_init=True, **kwargs
        )
        self.cfg = cfg
        h_dim = cfg.h_dim
        w_dim = cfg.w_dim
        z_dim = cfg.z_dim # dim of the query
        s_dim = cfg.s_dim

        assert cfg.z_dim >= 0

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        eval(f"self.{cfg.param_fn}_param")() # param. of binary rules 

        root_dim = s_dim
        root_modules = ( 
            nn.Linear(root_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, NT),
        ) 
        self.root_mlp = nn.Sequential(*root_modules)

        self.enc_emb = None
        if cfg.param_fn.startswith("select"):
            self._build_z_encoder(vocab)

        term_dim = s_dim
        term_modules = ( 
            nn.Linear(term_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, len(vocab)),
        ) 
        self.term_mlp = nn.Sequential(*term_modules)
        self._initialize()

        if cfg.tied_terms and isinstance(self.enc_emb, PartiallyFixedEmbedding):
            self.term_mlp[-1] = enc_emb

    def default_param(self):
        rule_dim = s_dim = self.cfg.s_dim
        self.p_emb = nn.Parameter(torch.randn(self.NT, rule_dim))
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)

    def a_b_c_param(self):
        # factorize A->BC params into Aa x Bb x Cc
        s_dim = self.cfg.s_dim
        self.p_emb = nn.Parameter(torch.randn(self.NT, s_dim))
        self.c_emb = nn.Parameter(torch.randn(self.T, s_dim))
        rule_dim = s_dim
        self.p_mlp = nn.Sequential(
            nn.Linear(rule_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, s_dim),
        )
        rule_dim = s_dim
        self.lc_mlp = nn.Sequential(
            nn.Linear(rule_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, s_dim),
        )
        self.rc_mlp = nn.Sequential(
            nn.Linear(rule_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, s_dim),
        )

    def select_a_b_c_param(self):
        self.a_b_c_param()
        s_dim = self.cfg.s_dim
        z_dim = self.cfg.z_dim 
        self.p_key = nn.Linear(s_dim, z_dim)
        self.lc_key = nn.Linear(s_dim, z_dim)
        self.rc_key = nn.Linear(s_dim, z_dim)

        self.p_rank, self.lc_rank, self.rc_rank = \
            self.cfg.p_rank, self.cfg.lc_rank, self.cfg.rc_rank 
        self.scale = 1 / math.sqrt(z_dim)

    def a_bc_param(self):
        # factorize A->BC params into Aa x Rr
        s_dim = self.cfg.s_dim
        self.p_emb = nn.Parameter(torch.randn(self.NT, s_dim))
        self.c_emb = nn.Parameter(torch.randn(self.NT_T ** 2, s_dim))
        rule_dim = s_dim
        self.p_mlp = nn.Sequential(
            nn.Linear(rule_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, s_dim),
        )
        rule_dim = s_dim
        self.uc_mlp = nn.Sequential(
            nn.Linear(rule_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, s_dim),
        )

    def select_a_bc_param(self):
        self.a_bc_param()
        s_dim = self.cfg.s_dim
        z_dim = self.cfg.z_dim 
        self.p_key = nn.Linear(s_dim, z_dim)
        self.uc_key = nn.Linear(s_dim, z_dim)

        self.p_rank, self.uc_rank = self.cfg.p_rank, self.cfg.uc_rank
        self.scale = 1 / math.sqrt(z_dim)

    def _build_z_encoder(self, vocab):
        cfg = self.cfg
        if cfg.z_dim <= 0:
            return
        if not cfg.wo_enc_emb:
            i_dim = self._build_embedder_from_default(vocab)
        else:
            if os.path.isfile(cfg.w2vec_file):
                i_dim = self._build_embedder_from_w2v(vocab)
            else:
                i_dim = self._build_embedder_from_mlm(vocab)
        k = 3 if "a_b_c" in self.cfg.param_fn else 2
        self.enc_out = nn.Linear(i_dim, cfg.z_dim * k)

    def encode_query(
        self, x, lengths, *args, sub_words=None, token_indice=None, 
        use_mean=False, max_pooling=True, enforce_sorted=False, tags=None, **kwargs
    ):
        # encode query for S->N and N->NN
        self.z = None
        if not self.cfg.param_fn.startswith("select"):
            return # do not need query
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
            self.z = torch.cat([mean, lvar], dim=-1)
            self.kl_ = None #self.kl(mean, lvar).sum(1)

    @staticmethod
    def select_prob(b, emb, emb_mlp, key_mlp, query, rank):
        emb = emb_mlp(emb) # (b, NT, s)
        emb = emb.unsqueeze(0).expand(b, -1, -1)
        logit = torch.bmm(key_mlp(emb), query.unsqueeze(-1))

        p_topk = soft_topk(logit, rank, noise=100)
        p_emb = emb * p_topk # back propagate to z
        p_row = p_emb.masked_select(p_topk == 1)
        p_row = p_row.view(b, rank, -1)

        q_dist = logit.squeeze(-1).softmax(-1)
        q_logp = logit.squeeze(-1).log_softmax(-1)

        N = emb.shape[1]
        kl = (q_dist * (q_logp + math.log(N))).sum(-1)
        he = (q_dist * q_logp).sum(-1)
        return p_emb, p_row, p_topk, kl

    def default_prob(self, b):
        p_emb = self.p_emb.unsqueeze(0).expand(
            b, self.NT, self.s_dim
        )
        rule_prob = F.log_softmax(self.rule_mlp(p_emb), -1)
        rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob

    def a_bc_prob(self, b):
        rule_prob = F.log_softmax(torch.matmul(
            self.p_mlp(self.p_emb), self.uc_mlp(self.c_emb).transpose(0, 1)
        ), -1).unsqueeze(0).expand(b, -1, -1)
        rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob

    def iterative_inv(self, mat, n_iter=6, self_init_option=""):
        identity = torch.eye(mat.size(-1), device=mat.device)
        key = mat

        # The entries of key are positive and ||key||_{\infty} = 1 due to softmax
        if self_init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||key||_1, of initialization of Z_0, leading to faster convergence.
            value = 1 / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None] * key.transpose(-1, -2)

        for _ in range(n_iter):
            key_value = torch.matmul(key, value)
            value = torch.matmul(
                0.25 * value,
                13 * identity
                - torch.matmul(key_value, 15 * identity - torch.matmul(key_value, 7 * identity - key_value)),
            )
        return value

    def select_a_bc_prob(self, b):
        p_z, uc_z = self.z.chunk(2, -1)

        p_emb, p_row, p_topk, p_kl = self.select_prob(
            b, self.p_emb, self.p_mlp, self.p_key, p_z, self.p_rank
        )

        uc_emb, uc_row, uc_topk, uc_kl = self.select_prob(
            b, self.c_emb, self.uc_mlp, self.uc_key, p_z, self.uc_rank
        )

        self.kl_ = uc_kl * 1.

        factor1 = F.softmax(torch.bmm(p_emb, uc_row.transpose(1, 2)) * self.scale, -1)
        factor2 = F.softmax(torch.bmm(p_row, uc_emb.transpose(1, 2)) * self.scale, -1)
        #core = F.softmax(torch.bmm(uc_row, p_row.transpose(1, 2)), -1)

        core = F.softmax(torch.bmm(p_row, uc_row.transpose(1, 2)) * self.scale, -1)
        core = self.iterative_inv(core.unsqueeze(1)).squeeze(1)
        core = F.softmax(core, -1)

        rule_prob = torch.bmm(
            torch.bmm(factor1, core), factor2
        ).clamp(min=1e-9).log().view(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob

    def a_b_c_prob(self, b):
        c_emb = torch.cat((self.p_emb, self.c_emb), 0)
        c_emb = (
            self.lc_mlp(c_emb).unsqueeze(1) *
            self.rc_mlp(c_emb).unsqueeze(0)
        ).view(-1, self.s_dim) # (NT, NT, s)
        rule_prob = F.log_softmax(torch.matmul(
            self.p_mlp(self.p_emb), c_emb.transpose(0, 1)
        ), -1).unsqueeze(0).expand(b, -1, -1)
        rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob

    def select_a_b_c_prob(self, b):
        p_z, lc_z, rc_z = self.z.chunk(3, -1)

        p_emb, p_row, p_topk, _ = self.select_prob(
            b, self.p_emb, self.p_mlp, self.p_key, p_z, self.p_rank
        )

        c_emb = torch.cat((self.p_emb, self.c_emb), 0)

        lc_emb, lc_row, lc_topk, _ = self.select_prob(
            b, c_emb, self.lc_mlp, self.lc_key, lc_z, self.lc_rank
        )
        rc_emb, rc_row, rc_topk, _ = self.select_prob(
            b, c_emb, self.rc_mlp, self.rc_key, rc_z, self.rc_rank
        )

        factor1 = F.softmax(torch.bmm(p_emb, lc_row.transpose(1, 2)), -1) # A x NT_b
        factor2 = F.softmax(torch.bmm(rc_row, lc_emb.transpose(1, 2)), -1) # NT_c x B
        factor3 = F.softmax(torch.bmm(p_row, rc_emb.transpose(1, 2)), -1) # NT_a x C

        cfactor = (
            factor2.unsqueeze(2).unsqueeze(-1) * # (b, NT_c, 1, B, 1)
            factor3.unsqueeze(1).unsqueeze(-2)   # (b, 1, NT_a, 1, C)
        ).view(b, -1, self.NT_T ** 2) # (b, NT_c x NT_a, B x C)

        c_row = rc_row.unsqueeze(2) * p_row.unsqueeze(1) # (b, s, NT_c, NT_a)
        core = F.softmax(torch.bmm(
            lc_row, c_row.view(b, self.s_dim, -1)
        ), -1) # (b, NT_b, NT_c x NT_a)

        rule_prob = torch.bmm(
            torch.bmm(factor1, core), cfactor
        ).clamp(min=1e-9).log().view(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob

    def forward(
        self, x, lengths, *args, sub_words=None, token_indice=None, tags=None, 
        use_mean=False, max_pooling=True, enforce_sorted=False, **kwargs
    ):
        """ x, lengths: words; sub_words, token_indice: sub-word 
        """
        b, n = x.shape[:2]
        self.encode_query(
            x, lengths, *args, sub_words=sub_words, token_indice=token_indice, tags=tags,
            use_mean=use_mean, max_pooling=max_pooling, enforce_sorted=enforce_sorted, **kwargs
        )

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob

        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(b, n, -1, -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob
        
        rules_fn = f"self.{self.cfg.param_fn}_prob"
        roots_ll, terms_ll, rules_ll = roots(), terms(), eval(rules_fn)(b)
        kl = getattr(self, "kl_", None)
        extra = {}
        return (terms_ll, rules_ll, roots_ll), kl, extra
