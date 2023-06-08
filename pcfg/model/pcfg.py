from omegaconf import OmegaConf
import os, re
import torch
import numpy as np
from torch import nn
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import data_parallel

from torch_struct import SentCFG

from ..util import extract_parses, Stats
from ..pcfg import build_pcfg_head
from ..module import PretrainedEncoder, PartiallyFixedEmbedding
from . import load_checkpoint

class XPCFG(nn.Module):
    def __init__(self, cfg, echo):
        super(XPCFG, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.meter_train = Stats()
        self.meter_infer = Stats()

    def forward(
        self, sentences, lengths, *args, token_indice=None, sub_words=None, **kwargs
    ):
        params, kl, losses = self.pcfg_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words
        )
        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, argmax_trees = extract_parses(spans, lengths.tolist(), inc=0)
        nll = -dist.partition
        kl = torch.zeros_like(nll) if kl is None else kl
        meter = self.meter_train if self.training else self.meter_infer
        meter(
            nll=nll.detach().sum().item(), kl=kl.detach().sum().item()
        )
        loss = (nll + 0).mean()
        return loss, (argmax_spans, argmax_trees)
    
    def tokenizer(self):
        return getattr(self.pcfg_head.enc_emb, "tokenizer", None) if hasattr(self.pcfg_head, "enc_emb") else None

    def encode_text(self, text):
        return self.pcfg_head.enc_emb(text)[-1]

    def batch_stats(self, vocab=None, **kwargs):
        return None

    def stats(self, num_sents, num_words): 
        pnorm = lambda : sum([p.norm(p=2) ** 2 for p in self.parameters() 
            if p is not None and p.requires_grad
        ]) ** 0.5 #.item() ** 0.5
        gnorm = lambda : sum([p.grad.norm(p=2) ** 2 for p in self.parameters() 
            if p is not None and p.grad is not None
        ]) ** 0.5 #.item() ** 0.5

        meter = self.meter_train.stats if self.training else self.meter_infer.stats
        ppl = np.exp(meter["nll"] / num_words)
        kl = meter["kl"] / num_sents
        bound = np.exp((meter["nll"] + meter["kl"]) / num_words)

        return f"|Param|: {pnorm():.2f} |GParam|: {gnorm():.2f} PPL: {ppl:.2f} KL: {kl:.4f} PPLBound {bound:.2f}"

    def reset(self):
        meter = self.meter_train if self.training else self.meter_infer
        meter.reset()
    
    def reduce_grad(optim_rate, sync=False):
        self.pcfg_head.reduce_grad(optim_rate, sync=sync)

    def collect_state_dict(self):
        return (
            self.pcfg_head.state_dict(), 
        )

    def report(self, gold_file=None):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return "" 
        else:
            return ""
    
    def build(self, vocab=None, vocab_zh=None, num_tag=0, **kwargs):
        tunable_params = dict()
        if self.cfg.eval:
            local_cfg, sds, vocab, _, num_tag = load_checkpoint(self.cfg, self.echo)
            pcfg_head_sd = sds[0]
            self.build_model(
                vocab, vocab_zh=vocab_zh, num_tag=num_tag
            )
            n_o, o_n = self.pcfg_head.from_pretrained(pcfg_head_sd, strict=True)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize pcfg encoder from `pcfg_head`{msg}.")
            tunable_params = {"vocab": vocab, "num_tag": num_tag}
        else:
            tunable_params = self.build_model(
                vocab, vocab_zh=vocab_zh, num_tag=num_tag
            )
        self.cuda(self.cfg.rank)
        return tunable_params
    
    def build_model(self, vocab, vocab_zh=None, num_tag=-1):
        pcfg = self.cfg.model.pcfg
        kwargs = {"NT": pcfg.num_state, "T": num_tag, "vocab": vocab}
        self.pcfg_head = build_pcfg_head(pcfg, **kwargs) 

        tunable_params = {
            f"pcfg_head.{k}": v for k, v in self.pcfg_head.named_parameters() if not k.startswith("enc_emb")
        } 
        # deal with enc_emb
        if not pcfg.wo_enc_emb or isinstance(self.pcfg_head.enc_emb, PartiallyFixedEmbedding) or \
            (isinstance(self.pcfg_head.enc_emb, PretrainedEncoder) and pcfg.fine_tuned):
            tunable_params.update({ 
                f"pcfg_head.{k}": v for k, v in self.pcfg_head.named_parameters() if k.startswith("enc_emb")
            })
        return tunable_params
