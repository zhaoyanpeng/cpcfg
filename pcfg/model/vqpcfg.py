from omegaconf import OmegaConf
import os, re
import torch
import numpy as np
from torch import nn
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import data_parallel

from torch_struct import SentCFG

from ..util import postprocess_parses, extract_parses, Stats
from ..pcfg import build_pcfg_head
from ..pcfg.algo import TDSeqCFG
from ..module import PretrainedEncoder, PartiallyFixedEmbedding
from . import XPCFG

class VQPCFG(XPCFG):
    def __init__(self, cfg, echo):
        super(VQPCFG, self).__init__(cfg, echo)

    def forward(
        self, sentences, lengths, *args, token_indice=None, sub_words=None, use_mean=False, **kwargs
    ):
        params, kl, extra = self.pcfg_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words, 
            use_mean=(not self.training), **kwargs
        )
        self.set_batch_stats(sentences, lengths, extra)
        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, argmax_trees = extract_parses(spans, lengths.tolist(), inc=0)
        nll = -dist.partition
        kl = torch.zeros_like(nll) if kl is None else kl
        kl = self.set_stats(nll, kl, extra)
        loss = (nll + kl).mean()
        return loss, (argmax_spans, argmax_trees)

    def set_batch_stats(self, sentences, lengths, extra):
        self.last_batch = {
            "sentences": sentences, "lengths": lengths, "extra": extra 
        }
        return None

    def batch_stats(self, vocab=None, **kwargs):
        extra = self.last_batch["extra"]
        indice = extra["indice"]
        stats = indice[0].tolist()
        return f"VQ Code: {stats}"

    def set_stats(self, nll, kl, extra):
        meter = self.meter_train if self.training else self.meter_infer
        losses = {
            "nll": nll.detach().sum().item(), "kl": kl.detach().sum().item(),
            "vq_code": extra["vq_code"].detach().sum(0) # accumulate VQ code
        }
        combined_loss = kl
        for k, v in extra.items():
            if not k.endswith("_loss"):
                continue
            assert kl.shape == v.shape
            combined_loss += v # accumulate losses
            k = re.sub("_loss$", "", k) 
            losses[k] = v.detach().sum().item()
        meter(**losses)
        return combined_loss 

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
        
        avg_probs = meter["vq_code"] / num_sents
        vq_ppl = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        msg = f"PPL: {ppl:.2f} KL: {kl:.4f} PPLBound {bound:.2f} VQ_PPL: {vq_ppl:.2f}" 
        
        # other losses summed over sentences by default 
        for k, v in meter.items():
            if k in {"nll", "kl", "vq_code"}:
                continue
            msg += f" {k.upper()}: {v / num_sents:.4f}"
        return f"|Param|: {pnorm():.2f} |GParam|: {gnorm():.2f} {msg}"
