from omegaconf import OmegaConf
import os, re
import torch
from collections import OrderedDict

__all__ = ["load_checkpoint", "load_pcfg_init"]

def load_checkpoint(cfg, echo):
    model_file = f"{cfg.model_root}/{cfg.model_name}/{cfg.model_file}"
    if not os.path.isfile(model_file):
        echo(f"Failed to load the checkpoint `{model_file}`")
        return (None,) * 5
    echo(f"Loading from {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")
    vocab = checkpoint["vocab"]
    local_cfg = checkpoint["cfg"]
    local_str = OmegaConf.to_yaml(local_cfg)
    if cfg.verbose:
        echo(f"Old configs:\n\n{local_str}")
    nmodule = len(checkpoint["model"])
    if nmodule == 1:
        (pcfg_head_sd,) = checkpoint["model"]
        num_tag = pcfg_head_sd["term_emb"].shape[0]
        return local_cfg, (pcfg_head_sd, None, None, None), vocab, None, num_tag
    elif nmodule == 4:
        (pcfg_head_sd, gold_head_sd, text_head_sd, loss_head_sd) = checkpoint["model"]
        num_tag = pcfg_head_sd["term_emb"].shape[0]
        return local_cfg, (pcfg_head_sd, gold_head_sd, text_head_sd, loss_head_sd), vocab, None, num_tag
    else:
        raise ValueError(f"I don't know how to parse the checkpoint: # module is {nmodule}.")

def load_pcfg_init(cfg, echo):
    model_file = f"{cfg.model_root}/{cfg.model.pcfg.model_init}"
    if not os.path.isfile(model_file):
        return None
    echo(f"Loading pcfg initialization from {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")
    if isinstance(checkpoint["model"], dict): # from emnlp init
        pcfg_head_sd = checkpoint["model"]["parser"] # hard-coded
    else:
        pcfg_head_sd = checkpoint["model"][0]
    return pcfg_head_sd
