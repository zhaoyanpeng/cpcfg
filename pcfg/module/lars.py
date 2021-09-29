import math
import torch

__all__ = ["exclude_bias_or_norm", "adjust_learning_rate", "LARS"]

def exclude_bias_or_norm(p):
    return p.ndim < 2 

def adjust_learning_rate(cfg, optimizer, dataloader, step):
    max_steps = cfg.epochs * len(dataloader)
    warmup_steps = 10 * len(dataloader)
    base_lr = cfg.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * cfg.lr_weight
    optimizer.param_groups[1]['lr'] = lr * cfg.lr_bias

class LARS(torch.optim.Optimizer):
    def __init__(
            self, params, lr, 
            weight_decay=0, 
            momentum=0.9, 
            eta=0.001,
            weight_decay_filter=None, 
            lars_adaptation_filter=None
        ):
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum,
            eta=eta, 
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0., torch.where(
                            update_norm > 0,
                            (g['eta'] * param_norm / update_norm), one
                        ), one
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

