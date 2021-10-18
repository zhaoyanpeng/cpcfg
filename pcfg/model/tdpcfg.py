from omegaconf import OmegaConf
import os, re
import torch
import numpy as np
from torch import nn
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import data_parallel

from ..util import postprocess_parses, extract_parses, Stats
from ..pcfg import build_pcfg_head
from ..pcfg.algo import TDSeqCFG
from ..module import PretrainedEncoder, PartiallyFixedEmbedding
from . import XPCFG

class TDPCFG(XPCFG):
    def __init__(self, cfg, echo):
        super(TDPCFG, self).__init__(cfg, echo)

    def forward(
        self, sentences, lengths, *args, token_indice=None, sub_words=None, **kwargs
    ):
        params, kl, losses = self.pcfg_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words
        )
        dist = TDSeqCFG()
        ll, argmax_spans = dist.partition(params, lengths, mbr=(not self.training)) #True) #
        argmax_spans, argmax_trees = ((None,) * 2 if argmax_spans is None else 
            postprocess_parses(argmax_spans, lengths.tolist())
        )
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        meter = self.meter_train if self.training else self.meter_infer
        meter(
            nll=nll.detach().sum().item(), kl=kl.detach().sum().item()
        )
        loss = (nll + kl).mean()
        return loss, (argmax_spans, argmax_trees)
