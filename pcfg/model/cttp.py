from omegaconf import OmegaConf
import os, re
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F 
from torch.nn.parallel import data_parallel

from torch_struct import SentCFG

from ..util import extract_parses, Stats
from ..pcfg import build_pcfg_head
from ..module import build_text_head, build_loss_head, PretrainedEncoder, PartiallyFixedEmbedding
from . import load_checkpoint, load_pcfg_init

class CTTP(nn.Module):
    def __init__(self, cfg, echo):
        super(CTTP, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.meter_main = Stats()
        self.meter_eval = Stats()
        self.meter_train = Stats()
        self.meter_infer = Stats()

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
            nll=nll.detach().sum().item(), kl=kl.detach().sum().item(), cst=cst_loss.detach().sum().item()
        )
        loss = (nll + kl + cst_loss).mean()
        #loss = (nll + kl).mean()
        #loss = (cst_loss).mean()
        return loss, (argmax_spans, argmax_trees)
    
    def forward_contrast(self, sentences, lengths, span_margs, gold_embs, token_indice=None, sub_words=None):
        gold_embs = self.gold_head(gold_embs, normalized=True) 
        span_embs = self.text_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words, normalized=False
        ) 
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
            span_marg = span_margs[:, k].softmax(-1).unsqueeze(-2)
            span_vect = torch.matmul(span_marg, span_vect).squeeze(-2)
            span_vect = F.normalize(span_vect) 
            loss = self.loss_head(embs, span_vect, normalized=True)
            matching_loss_matrix[:, k] = loss
        span_margs = span_margs.sum(-1)
        expected_loss = span_margs[:, : nstep] * matching_loss_matrix 
        expected_loss = expected_loss.sum(-1)
        return expected_loss
    
    def retrieval_eval(self, embs, span_embs, span_margs, lengths):
        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim) 
        span_vect = torch.stack(
            [span_embs[b][k - 1] for b, k in enumerate(mstep)], dim=0
        ) 
        span_marg = torch.softmax(torch.stack(
            [span_margs[b][k - 1] for b, k in enumerate(mstep)], dim=0
        ), -1).unsqueeze(-2)
        span_vect = torch.bmm(span_marg, span_vect).squeeze(-2)
        span_vect = F.normalize(span_vect)
        loss = self.loss_head(embs, span_vect, normalized=True)
        #loss = loss or torch.tensor([]) # could be none
        return loss

    def forward_unlabeled_contrast(self, sentences, lengths, span_margs, gold_embs, token_indice=None, sub_words=None):
        gold_embs = self.gold_head(gold_embs, normalized=True)
        span_embs = self.text_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words, normalized=False
        )
        span_embs = span_embs.squeeze(-2) # label dim
        fn = self.unlabeled_contrastive_loss if self.training else self.unlabeled_retrieval_eval
        return fn(gold_embs, span_embs, span_margs, lengths)

    def unlabeled_contrastive_loss(self, embs, span_embs, span_margs, lengths):
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
            #span_marg = span_margs[:, k].softmax(-1).unsqueeze(-2)
            #span_vect = torch.matmul(span_marg, span_vect).squeeze(-2)
            span_vect = F.normalize(span_vect)
            loss = self.loss_head(embs, span_vect, normalized=True)
            matching_loss_matrix[:, k] = loss
        span_margs = span_margs.sum(-1)
        expected_loss = span_margs[:, : nstep] * matching_loss_matrix
        expected_loss = expected_loss.sum(-1)
        return expected_loss

    def unlabeled_retrieval_eval(self, embs, span_embs, span_margs, lengths):
        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim)
        span_vect = torch.stack(
            [span_embs[b][k - 1] for b, k in enumerate(mstep)], dim=0
        )
        #span_marg = torch.softmax(torch.stack(
        #    [span_margs[b][k - 1] for b, k in enumerate(mstep)], dim=0
        #), -1).unsqueeze(-2)
        #span_vect = torch.bmm(span_marg, span_vect).squeeze(-2)
        span_vect = F.normalize(span_vect)
        loss = self.loss_head(embs, span_vect, normalized=True)
        #loss = loss or torch.tensor([]) # could be none
        return loss

    def stats_main(self, num_sents, num_words): 
        meter = self.meter_main.stats if self.training else self.meter_eval.stats
        num_sents, num_words = max(num_sents, 1), max(num_words, 1)
        ppl = np.exp(meter["nll"] / num_words)
        kl = meter["kl"] / num_sents
        bound = np.exp((meter["nll"] + meter["kl"]) / num_words)
        cst = meter["cst"] / num_sents
        return (
            f"|Param|: {self.pnorm():.2f} |GParam|: {self.gnorm():.2f} " +
            f"PPL: {ppl:.2f} KL: {kl:.4f}, PPLBound {bound:.2f} CST: {cst:.4f}"
        )

    def reset_main(self):
        meter = self.meter_main if self.training else self.meter_eval
        meter.reset()

    def report_main(self, gold_file=None, **kwargs):
        if (not dist.is_initialized() or dist.get_rank() == 0) and self.loss_head is not None:
            return self.loss_head.report(gold_file=gold_file, **kwargs)
        else:
            return ""

    def forward(
        self, sentences, lengths, *args, token_indice=None, sub_words=None, use_mean=False, **kwargs
    ):
        params, kl, losses = self.pcfg_head(
            sentences, lengths, token_indice=token_indice, sub_words=sub_words, use_mean=use_mean
        )
        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, argmax_trees = extract_parses(spans, lengths.tolist(), inc=0)
        nll = -dist.partition
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
    
    def stats(self, *args): 
        meter = self.meter_train.stats if self.training else self.meter_infer.stats
        num_sents = meter["num_sents"] 
        num_words = meter["num_words"]
        ppl = np.exp(meter["nll"] / num_words)
        kl = meter["kl"] / num_sents
        bound = np.exp((meter["nll"] + meter["kl"]) / num_words)
        return f"PPL: {ppl:.2f} KL: {kl:.4f} PPLBound {bound:.2f}"

    def reset(self):
        meter = self.meter_train if self.training else self.meter_infer
        meter.reset()
    
    def report(self, gold_file=None):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return "" 
        else:
            return ""
    
    def tokenizer(self):
        return getattr(self.pcfg_head.enc_emb, "tokenizer", None) if hasattr(self.pcfg_head, "enc_emb") else None

    def encode_text(self, text):
        return self.pcfg_head.enc_emb(text)[-1]

    def pnorm(self):
        return sum([p.norm(p=2) ** 2 for p in self.parameters() 
            if p is not None and p.requires_grad
        ]).item() ** 0.5
    
    def gnorm(self):
        return sum([p.grad.norm(p=2) ** 2 for p in self.parameters() 
            if p is not None and p.grad is not None
        ]).item() ** 0.5

    def reduce_grad(optim_rate, sync=False):
        self.pcfg_head.reduce_grad(optim_rate, sync=sync)

    def collect_state_dict(self):
        return (
            self.pcfg_head.state_dict(), 
            self.gold_head.state_dict() if self.gold_head is not None else {},
            self.text_head.state_dict() if self.text_head is not None else {},
            self.loss_head.state_dict() if self.loss_head is not None else {},
        )

    def count_rnd_consumed(self):
        return {
            "pcfg": 0 if self.pcfg_head is None else self.pcfg_head.num_rnd_consumed,
            "gold": 0 if self.gold_head is None else self.gold_head.num_rnd_consumed,
            "text": 0 if self.text_head is None else self.text_head.num_rnd_consumed,
        }

    def build(self, vocab=None, vocab_zh=None, num_tag=0, **kwargs):
        tunable_params = dict()
        if self.cfg.eval:
            pass
        else:
            tunable_params = self.build_model(
                vocab, vocab_zh=vocab_zh, num_tag=num_tag
            )
            pcfg_head_sd = load_pcfg_init(self.cfg, self.echo)
            if pcfg_head_sd is not None:
                n_o, o_n = self.pcfg_head.from_pretrained(pcfg_head_sd, strict=True)
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize pcfg encoder from `pcfg_head`{msg}.")
        self.cuda(self.cfg.rank)
        return tunable_params
    
    def build_model(self, vocab, vocab_zh=None, num_tag=-1):
        pcfg = self.cfg.model.pcfg
        kwargs = {"NT": pcfg.num_state, "T": num_tag, "vocab": vocab}
        self.pcfg_head = build_pcfg_head(pcfg, **kwargs) 
        
        emb_key = "enc_emb"
        tunable_params = {
            f"pcfg_head.{k}": v for k, v in self.pcfg_head.named_parameters() if not k.startswith(emb_key)
        } 
        # deal with enc_emb
        if not pcfg.wo_enc_emb or isinstance(self.pcfg_head.enc_emb, PartiallyFixedEmbedding) or \
            (isinstance(self.pcfg_head.enc_emb, PretrainedEncoder) and pcfg.fine_tuned):
            tunable_params.update({ 
                f"pcfg_head.{k}": v for k, v in self.pcfg_head.named_parameters() if k.startswith(emb_key)
            })

        if getattr(self.cfg.model, "text", None) is None:
            self.gold_head = self.text_head = self.loss_head = None
            return tunable_params

        self.gold_head = build_text_head(self.cfg.model.gold) # from which to learn
        kwargs = {"vocab": vocab, "enc_emb": self.pcfg_head.enc_emb if self.cfg.model.text.share_emb else None}
        self.text_head = build_text_head(self.cfg.model.text, **kwargs)
        self.loss_head = build_loss_head(self.cfg.model.loss)

        tunable_params.update({ 
            f"gold_head.{k}": v for k, v in self.gold_head.named_parameters()
        })
        tunable_params.update({ 
            f"text_head.{k}": v for k, v in self.text_head.named_parameters() 
            if not self.cfg.model.text.share_emb or not k.startswith(emb_key)
        })
        tunable_params.update({ 
            f"loss_head.{k}": v for k, v in self.loss_head.named_parameters() 
            if k in set(self.loss_head.tunable_param_names())
        })
        return tunable_params
