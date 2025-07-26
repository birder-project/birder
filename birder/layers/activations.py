from functools import partial

import torch
from torch import nn


def get_activation_module(name: str) -> type[nn.Module]:
    activation_dict = {
        "identity": nn.Identity,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "celu": nn.CELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "gelu_tanh": partial(nn.GELU, approximate="tanh"),
        "quick_gelu": QuickGELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "hard_sigmoid": nn.Hardsigmoid,
        "hard_swish": nn.Hardswish,
        "hard_tanh": nn.Hardtanh,
    }

    return activation_dict[name]  # type: ignore


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quick_gelu(x)
