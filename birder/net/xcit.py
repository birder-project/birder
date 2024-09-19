"""
XCiT, adapted from
https://github.com/facebookresearch/xcit/blob/main/xcit.py

"XCiT: Cross-Covariance Image Transformers", https://arxiv.org/abs/2106.09681
"""

# Reference license: Apache-2.0

import math
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.cait import ClassAttention


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim: int, dim: int, temperature: int = 10000) -> None:
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, B: int, H: int, W: int) -> torch.Tensor:
        mask = torch.ones(B, H, W).to(self.token_projection.weight.device)
        y_embed = mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = mask.cumsum(dim=2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos


class ConvPatchEmbed(nn.Module):
    def __init__(self, patch_size: Literal[8, 16], dim: int) -> None:
        super().__init__()
        if patch_size == 16:
            self.proj = nn.Sequential(
                Conv2dNormActivation(
                    3,
                    dim // 8,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
                Conv2dNormActivation(
                    dim // 8,
                    dim // 4,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
                Conv2dNormActivation(
                    dim // 4,
                    dim // 2,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
                Conv2dNormActivation(
                    dim // 2,
                    dim,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
            )

        elif patch_size == 8:
            self.proj = nn.Sequential(
                Conv2dNormActivation(
                    3,
                    dim // 4,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
                Conv2dNormActivation(
                    dim // 4,
                    dim // 2,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
                Conv2dNormActivation(
                    dim // 2,
                    dim,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.GELU,
                    inplace=None,
                ),
            )

        else:
            raise ValueError("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        H = x.shape[2]
        W = x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return (x, H, W)


class ClassAttentionBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, proj_drop: float, drop_path: float, eta: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.attn = ClassAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop)

        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        x_attn = torch.concat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        x = torch.concat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.concat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)

        return x


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.conv_bn_act = Conv2dNormActivation(
            in_features,
            in_features,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            groups=in_features,
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, stride=(1, 1), padding=padding, groups=out_features
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, N, C) = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv_bn_act(x)
        x = self.conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class XCA(nn.Module):
    """
    Cross-Covariance Attention (XCA)
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = F.normalize(q, dim=-1) * self.temperature
        k = F.normalize(k, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)  # pylint: disable=not-callable

        x = x.permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class XCABlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, proj_drop: float, drop_path: float, eta: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop)
        self.drop_path = StochasticDepth(drop_path, mode="row")

        self.norm2 = nn.LayerNorm(dim)
        self.local_mp = LPI(in_features=dim, out_features=dim, kernel_size=(3, 3))

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.local_mp(self.norm2(x), H, W))
        x = x + self.drop_path(self.gamma3 * self.mlp(self.norm3(x)))

        return x


class XCiT(BaseNet):
    default_size = 224

    # pylint: disable=too-many-branches
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        net_param = int(self.net_param)

        drop_rate = 0.0
        cls_attn_layers = 2
        mlp_ratio = 4.0
        qkv_bias = True
        if net_param == 0:
            # Nano, patch 16
            patch_size: Literal[8, 16] = 16
            embed_dim = 128
            depth = 12
            num_heads = 4
            eta = 1.0
            drop_path_rate = 0.0

        elif net_param == 1:
            # Nano, patch 8
            patch_size = 8
            embed_dim = 128
            depth = 12
            num_heads = 4
            eta = 1.0
            drop_path_rate = 0.0

        elif net_param == 2:
            # Tiny, patch 16
            patch_size = 16
            embed_dim = 192
            depth = 12
            num_heads = 4
            eta = 1.0
            drop_path_rate = 0.0

        elif net_param == 3:
            # Tiny, patch 8
            patch_size = 8
            embed_dim = 192
            depth = 12
            num_heads = 4
            eta = 1.0
            drop_path_rate = 0.0

        elif net_param == 4:
            # Small, patch 16
            patch_size = 16
            embed_dim = 384
            depth = 12
            num_heads = 8
            eta = 1.0
            drop_path_rate = 0.05

        elif net_param == 5:
            # Small, patch 8
            patch_size = 8
            embed_dim = 384
            depth = 12
            num_heads = 8
            eta = 1.0
            drop_path_rate = 0.05

        elif net_param == 6:
            # Medium, patch 16
            patch_size = 16
            embed_dim = 512
            depth = 24
            num_heads = 8
            eta = 1e-5
            drop_path_rate = 0.15

        elif net_param == 7:
            # Medium, patch 8
            patch_size = 8
            embed_dim = 512
            depth = 24
            num_heads = 8
            eta = 1e-5
            drop_path_rate = 0.15

        elif net_param == 8:
            # Large, patch 16
            patch_size = 16
            embed_dim = 768
            depth = 24
            num_heads = 16
            eta = 1e-5
            drop_path_rate = 0.25

        elif net_param == 9:
            # Large, patch 8
            patch_size = 8
            embed_dim = 768
            depth = 24
            num_heads = 16
            eta = 1e-5
            drop_path_rate = 0.3

        else:
            raise ValueError(f"net_param = {self.net_param} not supported")

        self.patch_embed = ConvPatchEmbed(patch_size=patch_size, dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule

        self.block1 = nn.ModuleList()
        for i in range(depth):
            self.block1.append(
                XCABlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    drop_path=dpr[i],
                    eta=eta,
                )
            )

        layers2 = []
        for _ in range(cls_attn_layers):
            layers2.append(
                ClassAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    drop_path=0.0,
                    eta=eta,
                )
            )

        layers2.append(nn.LayerNorm(embed_dim))
        self.block2 = nn.Sequential(*layers2)
        self.pos_embed = PositionalEncodingFourier(hidden_dim=32, dim=embed_dim)

        self.embedding_size = embed_dim
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)

            if isinstance(m, nn.Linear) is True and m.bias is not None:
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) is True:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        (x, H, W) = self.patch_embed(x)

        pos_encoding = self.pos_embed(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
        x = x + pos_encoding

        for blk in self.block1:
            x = blk(x, H, W)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.concat((cls_tokens, x), dim=1)
        x = self.block2(x)

        return x[:, 0]


registry.register_alias("xcit_nano16", XCiT, 0)
registry.register_alias("xcit_nano8", XCiT, 1)
registry.register_alias("xcit_tiny16", XCiT, 2)
registry.register_alias("xcit_tiny8", XCiT, 3)
registry.register_alias("xcit_small16", XCiT, 4)
registry.register_alias("xcit_small8", XCiT, 5)
registry.register_alias("xcit_medium16", XCiT, 6)
registry.register_alias("xcit_medium8", XCiT, 7)
registry.register_alias("xcit_large16", XCiT, 8)
registry.register_alias("xcit_large8", XCiT, 9)

registry.register_weights(
    "xcit_nano16_il-common",
    {
        "description": "XCiT nano patch16 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 11.5,
                "sha256": "aa99df518dc40c657939a840f951598a8a0b5513e328c2f11e882bcb01b64ca5",
            },
            "pt2": {
                "file_size": 14.3,
                "sha256": "988a96e2ae63cc2f7dc663c1fa37f478b76034e4935d4384d5cb89dae359b145",
            },
        },
        "net": {"network": "xcit_nano16", "tag": "il-common"},
    },
)
