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

from .pcfg import NaivePCFG
from .base import PCFG, ResLayer
from ..module import PretrainedEncoder, PartiallyFixedEmbedding

class LexiconPCFG(NaivePCFG):
    def __init__(self, cfg, NT=0, T=0, vocab=None, **kwargs): 
        super(LexiconPCFG, self).__init__(
            cfg, NT=NT, T=T, vocab=vocab, skip_init=True, **kwargs
        )
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

        tied_terms = cfg.tied_terms and isinstance(self.enc_emb, PartiallyFixedEmbedding)
        term_dim_o = w_dim if tied_terms else s_dim 

        term_dim = s_dim if cfg.share_term else s_dim + z_dim
        term_modules = ( 
            nn.Linear(term_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, term_dim_o) if term_dim_o != s_dim else nn.Identity(),
            nn.Linear(term_dim_o, len(vocab)),
        ) 
        self.term_mlp = nn.Sequential(*term_modules)
        self._count_rnd_consumed()
        self._initialize()

        self.excluded = []
        if tied_terms:
            self.term_mlp[-1] = self.enc_emb
            self.excluded = {"term_mlp\.3\.*", "enc_emb", "enc_rnn"}

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

    def from_pretrained(self, state_dict, excluded={}, strict=True):
        excluded = [] if strict else (self.excluded if len(excluded) == 0 else excluded)
        pattern = "|".join([f"^{m}\." for m in excluded])
        new_dict = self.state_dict()
        old_dict = {
            k: v for k, v in state_dict.items() if pattern == "" or not re.match(pattern, k)
        }

        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")

        self.load_state_dict(new_dict, strict=strict)
        return n_o, o_n

    def from_pretrained_transfer(
        self, state_dict, vocab_new, vocab_old, excluded={}, strict=True, unknown_init=None
    ):
        excluded = [] if strict else (self.excluded if len(excluded) == 0 else excluded)
        excluded = {"term_mlp\.4\.*", "enc_emb"} # account for dynamic vocabulary
        pattern = "|".join([f"^{m}\." for m in excluded])
        new_dict = self.state_dict()
        old_dict = {
            k: v for k, v in state_dict.items() if pattern == "" or not re.match(pattern, k)
        }

        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")

        self.load_state_dict(new_dict, strict=False)

        #emb1, mlp1 = self.enc_emb, self.term_mlp[-1]

        #print(emb1.tuned_weight[:2, :10])
        #print(mlp1.tuned_weight[:2, :10])
        #print((emb1.tuned_weight == mlp1.tuned_weight).all())
        #import sys; sys.exit(0)

        # load learned word embeddings
        cfg = self.cfg
        enc_emb_old = PartiallyFixedEmbedding(
            vocab_old, cfg.w2vec_file, word_dim=cfg.w_dim, out_dim=-1,
        )
        enc_emb_old_dict = {
            k.rsplit(".", 1)[-1]: v for k, v in state_dict.items() if re.match("enc_emb", k)
        }
        enc_emb_old.load_state_dict(enc_emb_old_dict, strict=True)

        # new words that are outside of the pre-trained ones (e.g., Glove)
        # but are contained in the learned ones (e.g., the loaded model)
        if unknown_init == "random":
            pass # randomly initialize the embeddings of new words
        elif unknown_init == "unknown":
            # set the emebddigns of new words to that of the unknown words
            unknown_word = "<unk>"
            wids_new = self.enc_emb.get_tunable_word_list()
            assert vocab_old.has(unknown_word)
            wid_old = vocab_old.idx(unknown_word)
            vec_old = enc_emb_old.get_word_vector(wid_old)
            #wid_new = vocab_new.idx(unknown_word)
            #vec_new = self.enc_emb.get_word_vector(wid_new)
            #print((vec_new == vec_old).all())
            for wid_new in wids_new:
                self.enc_emb.set_word_vector(wid_new, vec_old)
        else: # the standard way
            wids_new = self.enc_emb.get_tunable_word_list()
            for wid_new in wids_new:
                word = vocab_new(wid_new)
                if not vocab_old.has(word):
                    continue
                wid_old = vocab_old.idx(word)
                vec_old = enc_emb_old.get_word_vector(wid_old)
                self.enc_emb.set_word_vector(wid_new, vec_old)

        #print(emb1.tuned_weight[:2, :10])
        #print(mlp1.tuned_weight[:2, :10])
        #print((emb1.tuned_weight == mlp1.tuned_weight).all())
        #import sys; sys.exit(0)

        return n_o, o_n
