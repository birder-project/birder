"""
MobileViT v2, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilevit.py
and
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit_v2.py

Paper "Separable Self-attention for Mobile Vision Transformers",
https://arxiv.org/abs/2206.02680
"""

# Reference license: Apache-2.0 and Apple open source (see license at reference)

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.mobilenet_v2 import InvertedResidual


class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_drop: float,
        proj_drop: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        (query, key, value) = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)

        return out


class LinearTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop, bias=True)
        self.drop_path1 = StochasticDepth(drop_path, mode="row")

        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(
                embed_dim, int(embed_dim * mlp_ratio), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
            ),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Conv2d(
                int(embed_dim * mlp_ratio), embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
            ),
        )
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class MobileVitBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        transformer_dim: int,
        transformer_depth: int,
        mlp_ratio: float,
        patch_size: tuple[int, int],
        attn_drop: float,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv_kxk = Conv2dNormActivation(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            groups=channels,
            activation_layer=nn.SiLU,
        )
        self.conv_1x1 = nn.Conv2d(
            channels, transformer_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.transformer = nn.Sequential(
            *[
                LinearTransformerBlock(
                    embed_dim=transformer_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rate,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=transformer_dim)

        self.conv_proj = Conv2dNormActivation(
            transformer_dim,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.SiLU,
        )

        self.patch_size = patch_size
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.shape
        (patch_h, patch_w) = self.patch_size
        new_h = math.ceil(H / patch_h) * patch_h
        new_w = math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)

        # Unfold (feature map -> patches), [B, C, H, W] -> [B, C, P, N]
        C = x.shape[1]
        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # Fold (patches -> feature map), [B, C, P, N] --> [B, C, H, W]
        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w).permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)

        x = self.conv_proj(x)

        return x


# pylint: disable=invalid-name
class MobileViT_v2(DetectorBackbone):
    default_size = (256, 256)
    auto_register = True

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is not None, "must set net-param"
        assert self.config is None, "config not supported"
        width_factor = net_param
        width_factor_values = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert width_factor in width_factor_values, f"alpha = {width_factor} not supported"

        patch_size = (2, 2)
        attn_drop = 0.1
        depths = [2, 4, 3]
        expansion = 2
        stem_channels = int(32 * width_factor)
        channels: tuple[int, ...] = (64, 128, 256, 384, 512)
        channels = tuple([int(c * width_factor) for c in channels])  # pylint: disable=consider-using-generator
        dims: tuple[int, ...] = (128, 192, 256)
        dims = tuple([int(d * width_factor) for d in dims])  # pylint: disable=consider-using-generator

        self.stem = Conv2dNormActivation(
            self.input_channels,
            stem_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=nn.SiLU,
            bias=True,
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []

        stages["stage1"] = nn.Sequential(
            InvertedResidual(
                stem_channels,
                channels[0],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                expansion_factor=expansion,
                shortcut=False,
                activation_layer=nn.SiLU,
            ),
            InvertedResidual(
                channels[0],
                channels[1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                expansion_factor=expansion,
                shortcut=False,
                activation_layer=nn.SiLU,
            ),
            InvertedResidual(
                channels[1],
                channels[1],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                expansion_factor=expansion,
                shortcut=True,
                activation_layer=nn.SiLU,
            ),
        )
        return_channels.append(channels[1])

        for idx in range(2, len(channels)):
            stages[f"stage{idx}"] = nn.Sequential(
                InvertedResidual(
                    channels[idx - 1],
                    channels[idx],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    expansion_factor=expansion,
                    shortcut=False,
                    activation_layer=nn.SiLU,
                ),
                MobileVitBlock(
                    channels[idx],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    transformer_dim=dims[idx - 2],
                    transformer_depth=depths[idx - 2],
                    mlp_ratio=2,
                    patch_size=patch_size,
                    attn_drop=attn_drop,
                ),
            )
            return_channels.append(channels[idx])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = channels[-1]
        self.classifier = self.create_classifier()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Linear(embed_dim, self.num_classes, bias=False)


registry.register_weights(
    "mobilevit_v2_1_il-common",
    {
        "description": "MobileViT v2 with width multiplier of 1.0 trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 17.6,
                "sha256": "2b45b7f2ffe3dd129d9a7e9690d2dfd0f93ac60f24d118b920a51bcb950fd95e",
            }
        },
        "net": {"network": "mobilevit_v2", "net_param": 1, "tag": "il-common"},
    },
)
registry.register_weights(
    "mobilevit_v2_1.5_il-common",
    {
        "description": "MobileViT v2 with width multiplier of 1.5 trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 38.8,
                "sha256": "acd28c3ee653b62c69ad765c1d99827cea5051deb6dbdd7b9c8d7612782c86a3",
            }
        },
        "net": {"network": "mobilevit_v2", "net_param": 1.5, "tag": "il-common"},
    },
)
