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
from ..module import soft_topk
from ..module import VectorQuantizer, VectorQuantizerEMA 
from ..module import PretrainedEncoder, PartiallyFixedEmbedding

class VQPCFG(PCFG):
    def __init__(self, cfg, NT=0, T=0, vocab=None, skip_init=False, **kwargs): 
        super(VQPCFG, self).__init__(cfg, NT=NT, T=T, vocab=vocab, **kwargs)
        if skip_init: return #
        self.cfg = cfg
        h_dim = cfg.h_dim
        w_dim = cfg.w_dim
        z_dim = cfg.z_dim # dim of the query
        s_dim = cfg.s_dim

        assert cfg.z_dim >= 0

        nt_rule = cfg.nt_rule if cfg.vq_rule else NT
        if not cfg.gold_tag:
            self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        else:
            self.register_parameter("term_emb", None)
        self.nonterm_emb = nn.Parameter(torch.randn(nt_rule, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim
        rule_modules = (
            nn.Linear(rule_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, self.NT_T ** 2),
        )
        self.rule_mlp = nn.Sequential(*rule_modules)

        root_dim = s_dim
        root_modules = ( 
            nn.Linear(root_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
        ) 
        self.root_mlp = nn.Sequential(*root_modules)
        nt_root = cfg.nt_root if cfg.vq_root or cfg.share_vq_rule else NT
        self.root_out = nn.Linear(s_dim, nt_root)

        self._build_z_encoder(vocab)

        if not cfg.gold_tag:
            term_dim = s_dim
            term_modules = ( 
                nn.Linear(term_dim, s_dim),
                ResLayer(s_dim, s_dim),
                ResLayer(s_dim, s_dim),
                nn.Linear(s_dim, len(vocab)),
            ) 
            self.term_mlp = nn.Sequential(*term_modules)
        else:
            self.register_parameter("term_mlp", None)
        self._initialize()

        if cfg.tied_terms and isinstance(self.enc_emb, PartiallyFixedEmbedding) and self.term_mlp is not None:
            self.term_mlp[-1] = enc_emb

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
        # xW -> key
        if self.cfg.vq_root and not self.cfg.share_vq_rule:
            # this is a bit ambiguous: `vq_root' only means that we will select
            # a subset of root rules but not quantize the selective queries
            nt_root = cfg.nt_root if cfg.vq_root or cfg.share_vq_rule else NT
            self.root_key = nn.Linear(cfg.s_dim, cfg.z_dim * nt_root)
        else:
            self.register_parameter("root_key", None)
        if self.cfg.vq_rule:
            # we quantize the selective queries by default
            self.rule_key = nn.Linear(cfg.s_dim, cfg.z_dim)
        else:
            self.register_parameter("rule_key", None)
        # xW -> query
        n_z = int(cfg.vq_root and not self.cfg.share_vq_rule) + int(cfg.vq_rule)
        self.enc_out = nn.Linear(i_dim, cfg.z_dim * n_z)
        # quantized query
        if cfg.vq_quant > 0 and cfg.vq_decay > 0.:
            self.vqvae = VectorQuantizerEMA(cfg.vq_quant, cfg.z_dim, decay=cfg.vq_decay)
        elif cfg.vq_quant > 0:
            self.vqvae = VectorQuantizer(cfg.vq_quant, cfg.z_dim)
        else:
            self.register_parameter("vqvae", None)

    def _build_embedder_from_w2v(self, vocab):
        cfg = self.cfg
        self.enc_emb = PartiallyFixedEmbedding(
            vocab, cfg.w2vec_file, word_dim=cfg.w_dim, out_dim=-1,
        )
        self.enc_rnn = nn.LSTM(
            cfg.w_dim, cfg.h_dim, bidirectional=True, num_layers=1, batch_first=True
        )
        i_dim = cfg.h_dim * 2
        return i_dim

    def enc_lexicons(self, x, lengths, max_pooling=True, enforce_sorted=False):
        output = x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths.cpu(), batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h, attn_weights = output.max(1)
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
            attn_weights = None
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar, (attn_weights,)
    
    @staticmethod
    def gumbelize(w):
        gumbels = (
            -torch.empty_like(
                w, memory_format=torch.legacy_contiguous_format
            ).exponential_().log()
        )  # ~Gumbel(0,1)
        return w + gumbels

    def quantize(self, inputs):
        if self.vqvae is not None:
            vq_loss, rule_key, vq_ppl, encodings = self.vqvae(inputs)
            indice = (encodings == 1).nonzero(as_tuple=True)
            vq_idx = indice[1] # selected quantized vector
            return rule_key, vq_loss, vq_idx, vq_ppl, encodings
        return inputs, 0., [], 0., None

    def encode_query(
        self, x, lengths, *args, sub_words=None, token_indice=None, 
        use_mean=False, max_pooling=True, enforce_sorted=False, tags=None, **kwargs
    ):
        # encode query for S->N and N->NN
        self.z = None
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
        # split query for S->N and N->NN
        vq_loss, vq_idx, encodings = 0., [], torch.tensor(0.) 
        root_z, rule_z = (None, None)
        if self.z is not None:# select start rules & binary rules
            bb, ee = 0, self.z_dim
            if self.cfg.vq_root and not self.cfg.share_vq_rule:   
                root_z = self.z[:, bb:ee]
                bb = ee
                ee += self.z_dim
            if self.cfg.vq_rule: # only quantize the query for N->NN
                rule_z = self.z[:, bb:ee]
                rule_z, vq_loss, vq_idx, _, encodings = self.quantize(rule_z)
                bb = ee
                ee += self.z_dim
        return root_z, rule_z, vq_loss, vq_idx, encodings

    def forward(
        self, x, lengths, *args, sub_words=None, token_indice=None, tags=None, 
        use_mean=False, max_pooling=True, enforce_sorted=False, **kwargs
    ):
        """ x, lengths: words; sub_words, token_indice: sub-word 
        """
        b, n = x.shape[:2]
        root_z, rule_z, vq_loss, vq_idx, encodings = self.encode_query(
            x, lengths, *args, sub_words=sub_words, token_indice=token_indice, tags=tags,
            use_mean=use_mean, max_pooling=max_pooling, enforce_sorted=enforce_sorted, **kwargs
        )

        def pre_estimate_logits(*args):
            if self.cfg.cosine_logits:
                return [F.normalize(x, dim=-1) for x in args]
            return args

        def terms(tags=None):
            if tags is not None:
                term_prob = torch.empty((b, n, self.T), device=x.device).fill_(-1e8)
                term_prob.scatter_(2, tags.unsqueeze(-1), 0)
                return term_prob
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            )
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def roots(root_z, indice=None):
            root_emb = self.root_emb.expand(b, self.s_dim)
            root_emb = self.root_mlp(root_emb)
            root_prob = self.root_out(root_emb)

            topk_loss, extra = 0., tuple()
            if self.cfg.share_vq_rule and indice is not None:
                root_prob = torch.gather(root_prob, -1, indice)
            elif self.cfg.vq_root:
                root_key = self.root_key(root_emb).view(b, -1, self.z_dim)
                root_key, root_z = pre_estimate_logits(root_key, root_z)
                topk_logit = torch.bmm(root_key, root_z.unsqueeze(-1)).squeeze(-1)

                topk_logit_gumbel = self.gumbelize(topk_logit) # Gumbel + Kmax
                root_topk = soft_topk(topk_logit_gumbel, self.NT, noise=100)

                root_prob = root_prob * root_topk
                root_prob = root_prob.masked_select(root_topk == 1)
                root_prob = root_prob.view(b, -1)

                indice = (root_topk == 1).nonzero(as_tuple=True)
                indice = indice[1].view(b, -1) # selected subset

                _, old_indice = torch.topk(topk_logit, self.NT, largest=True, dim=1)
                old_indice, _ = old_indice.sort()
                extra = (indice,)

            root_prob = F.log_softmax(root_prob, -1)
            return root_prob, extra
        
        def rules(rule_z):
            rule_emb = self.nonterm_emb.unsqueeze(0).expand(b, -1, -1)

            topk_loss, extra = 0., (None,)
            if self.cfg.vq_rule:
                rule_key = self.rule_key(self.nonterm_emb)
                rule_key, rule_z = pre_estimate_logits(rule_key, rule_z)
                topk_logit = torch.matmul(rule_z, rule_key.transpose(0, 1))

                topk_logit_gumbel = self.gumbelize(topk_logit) # Gumbel + Kmax
                rule_topk = soft_topk(topk_logit_gumbel, self.NT, noise=100)

                rule_emb = rule_emb * rule_topk.unsqueeze(-1)
                rule_emb = rule_emb.masked_select(rule_topk.unsqueeze(-1) == 1)
                rule_emb = rule_emb.view(b, -1, self.s_dim)

                indice = (rule_topk == 1).nonzero(as_tuple=True)
                indice = indice[1].view(b, -1) # selected subset

                _, old_indice = torch.topk(topk_logit, self.NT, largest=True, dim=1)
                old_indice, _ = old_indice.sort()
                extra = (indice,)

            rule_prob = F.log_softmax(self.rule_mlp(rule_emb), -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob, extra

        terms_ll = terms(tags=tags)
        rules_ll, rule_extra = rules(rule_z)
        roots_ll, root_extra = roots(root_z, rule_extra[-1])

        vq_loss = torch.zeros((b,), device=x.device) + vq_loss
        extra = {"indice": rule_extra[0], "vq_loss": vq_loss, "vq_code": encodings}
        return (terms_ll, rules_ll, roots_ll), None, extra
