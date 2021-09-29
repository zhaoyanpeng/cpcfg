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

from ..module import PretrainedEncoder, PartiallyFixedEmbedding

PCFG_HEADS_REGISTRY = Registry("PCFG_HEADS")
PCFG_HEADS_REGISTRY.__doc__ = """
Registry for PCFG parameter encoders.
"""

def build_pcfg_head(cfg, **kwargs):
    return PCFG_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x) + x

@PCFG_HEADS_REGISTRY.register()
class NaivePCFG(torch.nn.Module):
    def __init__(self, cfg, NT=0, T=0, vocab=None, **kwargs): 
        super(NaivePCFG, self).__init__()
        self.cfg = cfg
        h_dim = cfg.h_dim
        w_dim = cfg.w_dim
        z_dim = cfg.z_dim
        s_dim = cfg.s_dim

        assert z_dim >= 0

        self.NT_T = NT + T
        self.NT = NT
        self.T = T

        self.z_dim = cfg.z_dim
        self.s_dim = cfg.s_dim

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
        self._initialize()

        if cfg.tied_terms and isinstance(self.enc_emb, PartiallyFixedEmbedding):
            self.term_mlp[-1] = enc_emb

    def _build_z_encoder(self, vocab):
        cfg = self.cfg
        if cfg.z_dim <= 0:
            return
        if not cfg.wo_enc_emb:
            self.enc_emb = nn.Embedding(len(vocab), cfg.w_dim)
            self.enc_rnn = nn.LSTM(
                cfg.w_dim, cfg.h_dim, bidirectional=True, num_layers=1, batch_first=True
            )
            i_dim = cfg.h_dim * 2
        else:
            if os.path.isfile(cfg.w2vec_file):
                self.enc_emb = PartiallyFixedEmbedding(
                    vocab, cfg.w2vec_file, word_dim=cfg.w_dim, out_dim=cfg.w_dim
                )
            else:
                self.enc_emb = PretrainedEncoder(
                    cfg.mlm_model, 
                    out_dim=cfg.mlm_out_dim, 
                    as_encoder=cfg.as_encoder, 
                    fine_tuned=cfg.fine_tuned, 
                    pooler_type=cfg.mlm_pooler
                ) 
            i_dim = self.enc_emb.output_dim
        self.enc_out = nn.Linear(i_dim, cfg.z_dim * 2)

    def _initialize(self):
        skip_enc_emb = not isinstance(self.enc_emb, nn.Embedding)
        for name, p in self.named_parameters():
            if skip_enc_emb and name.startswith("enc_emb"):
                continue
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            
    def reduce_grad(self, world_size, sync=True):
        for p in self.parameters():
            if p.grad is not None:
                if sync:
                    dist.all_reduce(p.grad.data)
                p.grad.data /= world_size

    def __setstate__(self, state):
        super(NaivePCFG, self).__setstate__(state)
        # Support loading old NaivePCFG checkpoints
        pass

    def from_pretrained(self, state_dict, strict=True):
        excluded = []
        new_dict = self.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}

        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")

        self.load_state_dict(new_dict, strict=strict)
        return n_o, o_n

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc_semantic(self, x, lengths, max_pooling=True, enforce_sorted=False):
        token_lm_loss, _, h = self.enc_emb(x)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        attn_weights = None # TODO 
        return mean, lvar, (attn_weights, token_lm_loss)

    def enc_lexicons(self, x, lengths, max_pooling=True, enforce_sorted=False):
        output = x_embbed = self.enc_emb(x)
        mask = torch.arange(lengths.max(), device=x.device)[None, :] >= lengths[:, None]
        if max_pooling:
            padding_value = float("-inf")
            output = output.masked_fill(mask.unsqueeze(-1), padding_value)
            h, attn_weights = output.max(1)
        else:
            padding_value = 0
            output = output.masked_fill(mask.unsqueeze(-1), padding_value)
            h = output.sum(1)[0] / lengths.unsqueze(-1)
            attn_weights = None
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        token_lm_loss = None # TODO 
        return mean, lvar, (attn_weights, token_lm_loss)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
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

    def enc_with_loss_mlm(self, x, lengths, max_pooling=True, enforce_sorted=False):
        token_lm_loss, output = self.enc_emb(x)
        
        x = x[0] # words
        mask = torch.arange(lengths.max(), device=x.device)[None, :] >= lengths[:, None]
         
        logits = self.decoder(output)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        word_lm_loss = loss_fct(
            logits.view(-1, logits.size(-1)), x.masked_fill(mask, -100).view(-1)
        )
        mask = ~ mask
        probs = logits.softmax(-1).view(-1, logits.size(-1))
        probs = torch.gather(probs, 1, x.view(-1, 1))
        probs = probs.view_as(x) * mask 

        mask = ~ mask
        if max_pooling:
            padding_value = float("-inf")
            output = output.masked_fill(mask.unsqueeze(-1), padding_value)
            h, attn_weights = output.max(1)
        else:
            padding_value = 0
            output = output.masked_fill(mask.unsqueeze(-1), padding_value)
            h = output.sum(1)[0] / lengths.unsqueze(-1)
            attn_weights = None
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar, (attn_weights, word_lm_loss, probs, token_lm_loss)

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

