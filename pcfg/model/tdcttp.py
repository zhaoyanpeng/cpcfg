from omegaconf import OmegaConf
import os, re
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F 
from torch.nn.parallel import data_parallel

from ..util import postprocess_parses, extract_parses, Stats
from ..pcfg import build_pcfg_head
from ..pcfg.algo import TDSeqCFG
from ..module import build_text_head, build_loss_head, PretrainedEncoder, PartiallyFixedEmbedding
from . import CTTP

class TDCTTP(CTTP):
    def __init__(self, cfg, echo):
        super(TDCTTP, self).__init__(cfg, echo)

    def forward_main(
        self, sentences, lengths, *args, token_indice=None, sub_words=None, gold_embs=None, **kwargs
    ):
        params, kl, losses = self.pcfg_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words
        )
        dist = TDSeqCFG()

        if gold_embs is None or gold_embs.shape[-1] == 0 or self.text_head is None:
            ll, argmax_spans = dist.partition(params, lengths, mbr=(not self.training)) #True) #
            nll = -ll
            cst_loss = None 
        else:
            ll, argmax_spans, span_margs = dist.partition(
                params, lengths, mbr=(not self.training), require_marginal=True
            )
            nll = -ll
            forward_contrast = self.forward_unlabeled_contrast if self.text_head.num_state == 1 else self.forward_contrast
            cst_loss = forward_contrast(
                sentences, lengths, span_margs, gold_embs, token_indice=token_indice, sub_words=sub_words
            )

        argmax_spans, argmax_trees = ((None,) * 2 if argmax_spans is None else 
            postprocess_parses(argmax_spans, lengths.tolist())
        )

        kl = torch.zeros_like(nll) if kl is None else kl
        cst_loss = torch.zeros_like(nll) if cst_loss is None else cst_loss
        meter = self.meter_main if self.training else self.meter_eval
        meter(
            nll=nll.detach().sum().item(), kl=kl.detach().sum().item(), cst=cst_loss.detach().sum().item()
        )
        loss = (nll + kl + cst_loss).mean()
        #loss = (nll + kl).mean()
        #loss = (cst_loss).mean()
        return loss, (argmax_spans, argmax_trees)

    def forward_contrast(self, sentences, lengths, span_margs, gold_embs, token_indice=None, sub_words=None):
        gold_embs = self.gold_head(gold_embs, normalized=True) 
        softmax = True
        if softmax:
            normalized_margs = span_margs.softmax(-1) # to average syntactic-type embeddings
        else:
            assert torch.all(span_margs >= 0), f"oops! there are negative marginals."
            normalizer = span_margs.sum(-1, keepdim=True)
            normalized_margs = span_margs / normalizer.masked_fill(normalizer == 0, 1.)
        span_embs = self.text_head(normalized_margs, normalized=True, pcfg_head=self.pcfg_head)
        fn = self.contrastive_loss if self.training else self.retrieval_eval
        return fn(gold_embs, span_embs, span_margs, lengths) 

    def contrastive_loss(self, embs, span_embs, span_margs, lengths):
        B = embs.shape[0]
        N = lengths.max()
        nstep = int(N * (N - 1) / 2)
        mstep = (lengths * (lengths - 1) / 2).int()
        # focus on only short spans
        nstep = int(mstep.float().mean().item() / 2)
        matching_loss_matrix = torch.zeros(
            B, nstep, device=embs.device
        )
        for k in range(nstep):
            span_vect = span_embs[:, k] 
            loss = self.loss_head(embs, span_vect, normalized=True)
            matching_loss_matrix[:, k] = loss
        span_margs = span_margs.sum(-1) # marginalize over syntactic types
        expected_loss = span_margs[:, : nstep] * matching_loss_matrix
        expected_loss = expected_loss.sum(-1)
        return expected_loss

    def retrieval_eval(self, embs, span_embs, span_margs, lengths):
        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim) 
        span_vect = torch.stack(
            [span_embs[b][k - 1] for b, k in enumerate(mstep)], dim=0
        ) 
        loss = self.loss_head(embs, span_vect, normalized=True)
        #loss = loss or torch.tensor([]) # could be none
        return loss
    
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

        bsize = sentences.shape[0]
        # we implicitly generate </s> so we explicitly count it
        num_words = (lengths + 1).sum().item() 

        meter = self.meter_train if self.training else self.meter_infer
        meter(
            num_sents=bsize, num_words=num_words,
            nll=nll.detach().sum().item(), kl=kl.detach().sum().item()
        )
        loss = (nll + kl).mean()
        return loss, (argmax_spans, argmax_trees)
    
