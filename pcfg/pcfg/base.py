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

class PCFG(torch.nn.Module):
    def __init__(self, cfg, NT=0, T=0, vocab=None, **kwargs): 
        super(PCFG, self).__init__()
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
        self._num_rnd_consumed = 0

    @property
    def num_rnd_consumed(self):
        return self._num_rnd_consumed

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)

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
        self.enc_out = nn.Linear(i_dim, cfg.z_dim * 2)

    def _build_embedder_from_default(self, vocab):
        cfg = self.cfg
        self.enc_emb = nn.Embedding(len(vocab), cfg.w_dim)
        self.enc_rnn = nn.LSTM(
            cfg.w_dim, cfg.h_dim, bidirectional=True, num_layers=1, batch_first=True
        )
        i_dim = cfg.h_dim * 2
        return i_dim

    def _build_embedder_from_w2v(self, vocab):
        cfg = self.cfg
        self.enc_emb = PartiallyFixedEmbedding(
            vocab, cfg.w2vec_file, word_dim=cfg.w_dim, out_dim=cfg.w_dim,
        )
        i_dim = self.enc_emb.output_dim
        return i_dim

    def _build_embedder_from_mlm(self, vocab):
        cfg = self.cfg
        self.enc_emb = PretrainedEncoder(
            cfg.mlm_model,
            out_dim=cfg.mlm_out_dim,
            as_encoder=cfg.as_encoder,
            fine_tuned=cfg.fine_tuned,
            pooler_type=cfg.mlm_pooler
        )
        i_dim = self.enc_emb.output_dim
        return i_dim

    def _initialize(self):
        skip_enc_emb = hasattr(self, "enc_emb") and not isinstance(self.enc_emb, nn.Embedding)
        for name, p in self.named_parameters():
            if skip_enc_emb and name.startswith("enc_emb"):
                continue
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
                self._num_rnd_consumed += torch.numel(p)

    def _count_rnd_consumed(self):
        for k, p in self.named_parameters():
            self._num_rnd_consumed += torch.numel(p)
            
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

    def from_pretrained(self, state_dict, excluded={}, strict=True):
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
        pass
