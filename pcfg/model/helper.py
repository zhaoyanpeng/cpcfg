from omegaconf import OmegaConf
import os, re
import torch
from collections import OrderedDict

__all__ = ["load_checkpoint"]

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
        return local_cfg, pcfg_head_sd, vocab, None, num_tag 
    else:
        raise ValueError(f"I don't know how to parse the checkpoint: # module is {nmodule}.")

