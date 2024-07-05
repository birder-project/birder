"""
CaiT, adapted from
https://github.com/facebookresearch/deit/blob/main/cait_models.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cait.py

"Going deeper with Image Transformers", https://arxiv.org/abs/2103.17239
"""

# Reference license: Apache-2.0 and Apache-2.0

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.core.net.base import BaseNet
from birder.core.net.vit import adjust_position_embedding


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: tuple[int, int]) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

        return x


class ClassAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_cls = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # pylint: disable=not-callable

        x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


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

    def forward(self, x: torch.Tensor, x_cls: torch.Tensor) -> torch.Tensor:
        u = torch.concat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma2 * self.mlp(self.norm2(x_cls)))

        return x_cls


class TalkingHeadAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float) -> None:
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0] * self.scale
        k = qkv[1]
        v = qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: float,
        init_values: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TalkingHeadAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CaiT(BaseNet):
    default_size = 224

    # pylint: disable=too-many-statements
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
        attn_drop = 0.0
        cls_attn_layers = 2
        mlp_ratio = 4.0
        qkv_bias = True
        if net_param == 0:
            # xxs 24
            patch_size = (16, 16)
            embed_dim = 192
            depth = 24
            num_heads = 4
            init_values = 1e-5
            drop_path_rate = 0.05

        elif net_param == 1:
            # xxs 36
            patch_size = (16, 16)
            embed_dim = 192
            depth = 36
            num_heads = 4
            init_values = 1e-5
            drop_path_rate = 0.1

        elif net_param == 2:
            # xs 24
            patch_size = (16, 16)
            embed_dim = 288
            depth = 24
            num_heads = 6
            init_values = 1e-5
            drop_path_rate = 0.05

        elif net_param == 3:
            # s 24
            patch_size = (16, 16)
            embed_dim = 384
            depth = 24
            num_heads = 8
            init_values = 1e-5
            drop_path_rate = 0.1

        elif net_param == 4:
            # s 36
            patch_size = (16, 16)
            embed_dim = 384
            depth = 36
            num_heads = 8
            init_values = 1e-6
            drop_path_rate = 0.2

        elif net_param == 5:
            # m 36
            patch_size = (16, 16)
            embed_dim = 768
            depth = 36
            num_heads = 16
            init_values = 1e-6
            drop_path_rate = 0.3

        elif net_param == 6:
            # m 48
            patch_size = (16, 16)
            embed_dim = 768
            depth = 48
            num_heads = 16
            init_values = 1e-6
            drop_path_rate = 0.4

        else:
            raise ValueError(f"net_param = {self.net_param} not supported")

        self.patch_size = patch_size = (16, 16)
        self.patch_embed = PatchEmbed(self.input_channels, embed_dim, patch_size)
        num_patches = (self.size // patch_size[0]) * (self.size // patch_size[1])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for _ in range(depth)]  # Uniform stochastic depth

        layers1 = []
        for i in range(depth):
            layers1.append(
                LayerScaleBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    init_values=init_values,
                )
            )

        self.block1 = nn.Sequential(*layers1)

        self.block2 = nn.ModuleList()
        for _ in range(cls_attn_layers):
            self.block2.append(
                ClassAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    drop_path=0.0,
                    eta=init_values,
                )
            )

        self.norm = nn.LayerNorm(embed_dim)

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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.block1(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for blk in self.block2:
            cls_tokens = blk(x, cls_tokens)

        x = torch.concat((cls_tokens, x), dim=1)
        x = self.norm(x)

        return x[:, 0]

    def create_classifier(self) -> nn.Module:
        return nn.Linear(self.embedding_size, self.num_classes)

    def adjust_size(self, new_size: int) -> None:
        super().adjust_size(new_size)

        # Sort out sizes
        num_pos_tokens = self.pos_embed.shape[1]
        num_new_tokens = (new_size // self.patch_size[0]) * (new_size // self.patch_size[1])
        if num_new_tokens == num_pos_tokens:
            return

        # Add back class tokens
        self.pos_embed = nn.Parameter(
            adjust_position_embedding(num_pos_tokens, self.pos_embed, new_size // self.patch_size[0], 0)
        )

        logging.info(f"Resized position embedding: {num_pos_tokens} to {num_new_tokens}")