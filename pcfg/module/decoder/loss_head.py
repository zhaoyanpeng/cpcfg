from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from collections import defaultdict

LOSS_HEADS_REGISTRY = Registry("LOSS_HEADS")
LOSS_HEADS_REGISTRY.__doc__ = """
Registry for image encoders.
"""

def build_loss_head(cfg, **kwargs):
    return LOSS_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

class LossHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = False
        self.normalized = True

    def tunable_param_names(self):
        return [k for k, v in self.named_parameters() if v.requires_grad]

    def infer(self, x1, x2, *args, **kwargs):
        if not hasattr(self, "x1s") or not hasattr(self, "x2s") or not hasattr(self, "ids"): 
            self.x1s, self.x2s, self.ids = [], [], []
        # normalized features
        if not kwargs.get("normalized", False):
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            x2 = x2 / x2.norm(dim=-1, keepdim=True)
        self.x1s.append(x1)
        self.x2s.append(x2)
        names = kwargs.get("names", None)
        if names is not None:
            self.ids.extend(names)
        return None  

    @staticmethod
    def retrieval_metrics(ranks, nsample=None, msg=""):
        nsample = nsample or ranks.shape[0]
        R1 = torch.where(ranks < 1)[0].shape[0] / nsample * 100.
        R5 = torch.where(ranks < 5)[0].shape[0] / nsample * 100.
        R10 = torch.where(ranks < 10)[0].shape[0] / nsample * 100.
        R50 = torch.where(ranks < 50)[0].shape[0] / nsample * 100.
        MED = ranks.median() + 1
        AVG = ranks.mean() + 1
        msg = f"{msg}: R@1 {R1:2.2f} R5 {R5:2.2f} R10 {R10:2.2f} R50 {R50:2.2f} MED {MED:2.2f} AVG {AVG:2.2f}"
        return msg

    @staticmethod
    def retrieval_eval(x1s, x2s, k=5):
        # assume x1s.shape[0] * k == x2s.shape[0]
        # x1 -> x2
        x12 = x1s @ x2s.t()
        nsample = x1s.shape[0]
        ind_12 = x12.argsort(descending=True)
        ranks = torch.zeros(nsample, device=x1s.device)
        for i in range(nsample):
            rank = 1e9
            inds = ind_12[i : i + 1]
            for j in range(i * k, i * k + k):
                rank_j = torch.where(inds == j)[1][0]
                if rank_j < rank:
                    rank = rank_j
            ranks[i] = rank
        msg_12 = LossHead.retrieval_metrics(ranks, msg="I->X")

        # x2 -> x1
        x21 = x2s @ x1s.t()
        nsample = x1s.shape[0]
        ind_21 = x21.argsort(descending=True)
        ranks = torch.zeros(nsample * k, device=x1s.device)
        for i in range(nsample):
            inds = ind_21[i * k : i * k + k]
            for j in range(k):
                ranks[i * k + j] = torch.where(inds[j : j + 1] == i)[1][0]
        msg_21 = LossHead.retrieval_metrics(ranks, msg="X->I")
        return f"{msg_12}\n{msg_21}"

    def report(self, gold_file=None, num_x2_per_x1=1, **kwargs):
        x1s = torch.cat(self.x1s)
        x2s = torch.cat(self.x2s)

        nsample = x1s.shape[0]
        labels = torch.arange(nsample, device=x1s.device).unsqueeze(-1)
        # x1 -> x2
        x12 = x1s @ x2s.t()
        ind_12 = x12.argsort(descending=True)
        r12 = torch.where(ind_12 == labels)[1]
        
        t12_1 = torch.where(r12 < 1)[0].shape[0] / nsample * 100. 
        t12_5 = torch.where(r12 < 5)[0].shape[0] / nsample * 100. 

        p_12 = f"I->X: t1 = {t12_1:2.2f} t5 = {t12_5:2.2f}" 

        # x2 -> x1
        x21 = x2s @ x1s.t()
        ind_21 = x21.argsort(descending=True)
        r21 = torch.where(ind_21 == labels)[1]

        t21_1 = torch.where(r21 < 1)[0].shape[0] / nsample * 100. 
        t21_5 = torch.where(r21 < 5)[0].shape[0] / nsample * 100. 

        p_21 = f"X->I: t1 = {t21_1:2.2f} t5 = {t21_5:2.2f}" 

        # mscoco retrieval
        indice = torch.arange(
            0, x1s.shape[0], num_x2_per_x1, device=x1s.device
        )
        x1s = x1s[indice]
        ref_metric = self.retrieval_eval(x1s, x2s, k=num_x2_per_x1)

        del self.x1s, self.x2s, self.ids
        ref = "" if ref_metric == "" else f"\nREFERENCE\n{ref_metric}"
        report = f"{p_12} {p_21} @ {x1s.shape[0]}{ref}"
        return report

@LOSS_HEADS_REGISTRY.register()
class CELossHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.logit_scale = (
            nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if cfg.scaling else
            torch.ones([], requires_grad=False) * np.log(1 / 1)
        )
        self.scale_max = cfg.scale_max or float("inf")
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambd_cst = cfg.lambd_cst
        self.reduce = False 
    
    def copy_state_dict(self, state_dict): 
        key = "logit_scale"
        new_dict = self.state_dict()
        new_dict.update({key: state_dict[key]})
        self.load_state_dict(new_dict)

    def forward(self, x1, x2, *args, **kwargs):
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                return self.infer(x1, x2, *args, **kwargs)
            return None 
        # normalized features
        if not kwargs.get("normalized", False):
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            x2 = x2 / x2.norm(dim=-1, keepdim=True)
        #print(f"{threading.current_thread().ident} loss --{kwargs.get('normalized', False)}")
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp().clamp(max=self.scale_max)
        logits_per_x1 = logit_scale * x1 @ x2.t()
        logits_per_x2 = logit_scale * x2 @ x1.t()
        # cross entropy loss 
        labels = torch.arange(x1.shape[0], device=x1.device)
        loss_mean_x1 = self.loss_fn(logits_per_x1, labels)
        loss_mean_x2 = self.loss_fn(logits_per_x2, labels)
        loss = (loss_mean_x1 + loss_mean_x2) * self.lambd_cst
        return loss

@LOSS_HEADS_REGISTRY.register()
class HingeLossHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.lambd_cst = cfg.lambd_cst
        self.margin = cfg.margin
        self.eps = cfg.eps
        self.reduce = False 
    
    def copy_state_dict(self, state_dict): 
        pass

    def forward(self, x1, x2, *args, **kwargs):
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                return self.infer(x1, x2, *args, **kwargs)
            return None 
        # normalized features
        if not kwargs.get("normalized", False):
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            x2 = x2 / x2.norm(dim=-1, keepdim=True)
        #print(f"{threading.current_thread().ident} loss --{kwargs.get('normalized', False)}")
        # cosine similarity as scores
        scores = x1 @ x2.t()
        diagonal = scores.diag().view(x1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_x1 = (self.margin + scores - d1).clamp(min=self.eps)
        loss_x2 = (self.margin + scores - d2).clamp(min=self.eps)
        I = torch.eye(scores.size(0), device=x1.device) > .5
        loss_x1 = loss_x1.masked_fill_(I, 0)
        loss_x2 = loss_x2.masked_fill_(I, 0)
        loss_mean_x1 = loss_x1.mean(1)
        loss_mean_x2 = loss_x2.mean(0)
        loss = (loss_mean_x1 + loss_mean_x2) * self.lambd_cst
        return loss
