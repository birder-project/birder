from collections.abc import Callable

import torch
from torch import nn
from torchvision.ops import MLP


class FFN(MLP):
    """
    Just a simple adaptor for interface consistency
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_features,
            [hidden_features, in_features],
            activation_layer=act_layer,
            inplace=None,
            bias=bias,
            dropout=dropout,
        )


# pylint: disable=invalid-name
class SwiGLU_FFN(nn.Module):
    """
    Paper "GLU Variants Improve Transformer", https://arxiv.org/abs/2002.05202
    Adapted from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop2 = nn.Dropout(dropout)

        # Weight initialization
        nn.init.normal_(self.fc1_g.weight, std=1e-6)
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
