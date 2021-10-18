from .base import build_pcfg_head, PCFG_HEADS_REGISTRY
from .pcfg import NaivePCFG
from .tdpcfg import TDPCFG

PCFG_HEADS_REGISTRY.register(TDPCFG)
PCFG_HEADS_REGISTRY.register(NaivePCFG)
