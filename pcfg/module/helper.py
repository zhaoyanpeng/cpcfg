import torch
from torch import nn

def layernorm_linear(sizes, layer_norm=False, ibias=True, bias=True):
    layers = list()
    for i in range(len(sizes) - 2):
        layers.extend([
            LayerNorm(sizes[i]),
            nn.Linear(sizes[i], sizes[i + 1], bias=ibias),
        ] if layer_norm else [
            nn.Linear(sizes[i], sizes[i + 1], bias=ibias),
        ])
    layers.extend([
        LayerNorm(sizes[-2]),
        nn.Linear(sizes[-2], sizes[-1], bias=bias),
    ] if layer_norm else [
        nn.Linear(sizes[-2], sizes[-1], bias=bias),
    ])
    return layers

def linear_relu_linear(sizes, layer_norm=False, ibias=True, bias=True):
    layers = [LayerNorm(sizes[0])] if layer_norm else list()
    for i in range(len(sizes) - 2):
        layers.extend([
            nn.Linear(sizes[i], sizes[i + 1], bias=ibias),
            LayerNorm(sizes[i + 1]),
            nn.ReLU(),
        ] if layer_norm else [
            nn.Linear(sizes[i], sizes[i + 1], bias=ibias),
            nn.ReLU(),
        ])
    layers.extend([
        nn.Linear(sizes[-2], sizes[-1], bias=bias),
    ])
    return layers

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
