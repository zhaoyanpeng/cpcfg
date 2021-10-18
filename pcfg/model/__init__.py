from .helper import *
from .pcfg import XPCFG
from .cttp import CTTP
from .tdpcfg import TDPCFG

from fvcore.common.registry import Registry

PARSER_MODELS_REGISTRY = Registry("PARSER_MODELS")
PARSER_MODELS_REGISTRY.__doc__ = """
Registry for parser models.
"""

def build_main_model(cfg, echo):
    return PARSER_MODELS_REGISTRY.get(cfg.worker)(cfg, echo)

PARSER_MODELS_REGISTRY.register(XPCFG)
PARSER_MODELS_REGISTRY.register(CTTP)
PARSER_MODELS_REGISTRY.register(TDPCFG)
