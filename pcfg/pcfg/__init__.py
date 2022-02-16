from .base import build_pcfg_head, PCFG_HEADS_REGISTRY
from .pcfg import NaivePCFG
from .tdpcfg import TDPCFG
from .vqpcfg import VQPCFG
from .lexicon_pcfg import LexiconPCFG
from .nystrom_pcfg import NystromPCFG

PCFG_HEADS_REGISTRY.register(TDPCFG)
PCFG_HEADS_REGISTRY.register(VQPCFG)
PCFG_HEADS_REGISTRY.register(NaivePCFG)
PCFG_HEADS_REGISTRY.register(LexiconPCFG)
PCFG_HEADS_REGISTRY.register(NystromPCFG)
