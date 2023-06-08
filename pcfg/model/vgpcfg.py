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
from . import CTTP, load_checkpoint, load_pcfg_init

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
            cst=cst_loss.detach().sum().item() / (
                1. if self.loss_head is None or self.loss_head.lambd_cst == 0. else self.loss_head.lambd_cst
            )
        )
        loss = (nll + kl + cst_loss).mean()
        #loss = (nll + kl).mean()
        #loss = (cst_loss).mean()
        return loss, (argmax_spans, argmax_trees)

    def build(self, vocab=None, vocab_zh=None, num_tag=0, **kwargs):
        tunable_params = dict()
        transfer_eval = getattr(self.cfg, "transfer_eval", None)
        if transfer_eval is not None:
            local_cfg, sds, vocab_old, _, num_tag = load_checkpoint(self.cfg, self.echo)
            pcfg_head_sd, gold_head_sd, text_head_sd, loss_head_sd = sds
            self.build_model(
                vocab, vocab_zh=vocab_zh, num_tag=num_tag
            )
            n_o, o_n = self.pcfg_head.from_pretrained_transfer(
                pcfg_head_sd, vocab, vocab_old, strict=True, unknown_init=transfer_eval
            )
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize `{transfer_eval}` pcfg encoder from `pcfg_head`{msg}.")
            tunable_params = {"vocab": vocab, "num_tag": num_tag}
            # TODO overide gold head, text head, and loss head
        elif self.cfg.eval: # the vocab is overriden by the loaded one
            local_cfg, sds, vocab, _, num_tag = load_checkpoint(self.cfg, self.echo)
            pcfg_head_sd, gold_head_sd, text_head_sd, loss_head_sd = sds
            self.build_model(
                vocab, vocab_zh=vocab_zh, num_tag=num_tag
            )
            n_o, o_n = self.pcfg_head.from_pretrained(pcfg_head_sd, strict=True)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize pcfg encoder from `pcfg_head`{msg}.")
            tunable_params = {"vocab": vocab, "num_tag": num_tag}
            # TODO overide gold head, text head, and loss head
        else:
            tunable_params = self.build_model(
                vocab, vocab_zh=vocab_zh, num_tag=num_tag
            )
            pcfg_head_sd = load_pcfg_init(self.cfg, self.echo)
            if pcfg_head_sd is not None:
                n_o, o_n = self.pcfg_head.from_pretrained(
                    pcfg_head_sd, excluded=list(self.cfg.model.pcfg.excluded), strict=False
                )
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize pcfg encoder from `pcfg_head`{msg}.")
        self.cuda(self.cfg.rank)
        return tunable_params
