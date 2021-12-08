from omegaconf import OmegaConf
import os, re
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F 
from torch.nn.parallel import data_parallel

from torch_struct import SentCFG

from ..util import postprocess_parses, extract_parses, Stats
from ..pcfg import build_pcfg_head
from ..pcfg.algo import TDSeqCFG
from ..module import build_text_head, build_loss_head, PretrainedEncoder, PartiallyFixedEmbedding
from . import CTTP

class VGPCFG(CTTP):
    def __init__(self, cfg, echo):
        super(VGPCFG, self).__init__(cfg, echo)

    def forward_main(
        self, sentences, lengths, *args, token_indice=None, sub_words=None, gold_embs=None, **kwargs
    ):
        params, kl, losses = self.pcfg_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words
        )
        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, argmax_trees = extract_parses(spans, lengths.tolist(), inc=0)

        if gold_embs is None or gold_embs.shape[-1] == 0 or self.text_head is None:
            nll = -dist.partition
            cst_loss = None 
        else:
            ll, span_margs = dist.inside_im
            nll = -ll
            forward_contrast = self.forward_unlabeled_contrast if self.text_head.num_state == 1 else self.forward_contrast
            cst_loss = forward_contrast(
                sentences, lengths, span_margs, gold_embs, token_indice=token_indice, sub_words=sub_words
            )

        kl = torch.zeros_like(nll) if kl is None else kl
        cst_loss = torch.zeros_like(nll) if cst_loss is None else cst_loss
        meter = self.meter_main if self.training else self.meter_eval
        meter(
            nll=nll.detach().sum().item(), kl=kl.detach().sum().item(), 
            cst=cst_loss.detach().sum().item() / (1. if self.loss_head is None else self.loss_head.lambd_cst)
        )
        loss = (nll + kl + cst_loss).mean()
        #loss = (nll + kl).mean()
        #loss = (cst_loss).mean()
        return loss, (argmax_spans, argmax_trees)
