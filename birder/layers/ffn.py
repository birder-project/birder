from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features, eps=norm_eps)
        else:
            self.norm = nn.Identity()

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
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class SoftMoE_FFN(nn.Module):
    """
    Paper "From Sparse to Soft Mixtures of Experts", https://arxiv.org/abs/2308.00951
    Adapted from: https://github.com/lucidrains/soft-moe-pytorch
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
        dropout: float = 0.0,
        num_experts: int = 4,
        num_slots: int = 1,
        normalize: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.slot_embeds = nn.Parameter(torch.empty(num_experts, num_slots, in_features).normal_(std=0.02))
        self.experts = nn.ModuleList(
            [
                FFN(in_features, hidden_features, act_layer=act_layer, bias=bias, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        if normalize is True:
            self.token_norm = nn.RMSNorm(in_features, eps=norm_eps)
            self.slot_norm = nn.RMSNorm(in_features, eps=norm_eps)
        else:
            self.token_norm = nn.Identity()
            self.slot_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route_x = self.token_norm(x)
        route_slots = self.slot_norm(self.slot_embeds)
        logits = torch.einsum("b n d, e s d -> b n e s", route_x, route_slots)

        dispatch_weights = F.softmax(logits, dim=1)
        combine_weights = F.softmax(logits.reshape(logits.size(0), logits.size(1), -1), dim=-1)
        combine_weights = combine_weights.reshape_as(logits)

        slots = torch.einsum("b n d, b n e s -> b e s d", x, dispatch_weights)
        expert_out = torch.stack([expert(slots[:, idx]) for idx, expert in enumerate(self.experts)], dim=1)
        x = torch.einsum("b e s d, b n e s -> b n d", expert_out, combine_weights)

        return x
