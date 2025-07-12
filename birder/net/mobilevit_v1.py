"""
MobileViT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilevit.py
and
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mobile_vit.py

Paper "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer",
https://arxiv.org/abs/2110.02178
"""

# Reference license: Apache-2.0 and MIT

import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.mobilenet_v2 import InvertedResidual
from birder.net.vit import EncoderBlock


class MobileVitBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        transformer_dim: int,
        transformer_depth: int,
        mlp_dim: int,
        patch_size: tuple[int, int],
        num_heads: int,
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
            activation_layer=nn.SiLU,
        )
        self.conv_1x1 = nn.Conv2d(
            channels, transformer_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.transformer = nn.Sequential(
            *[
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=transformer_dim,
                    mlp_dim=mlp_dim,
                    dropout=drop,
                    attention_dropout=attn_drop,
                    drop_path=drop_path_rate,
                    activation_layer=nn.SiLU,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.norm = nn.LayerNorm(transformer_dim)

        self.conv_proj = Conv2dNormActivation(
            transformer_dim,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.SiLU,
        )
        self.conv_fusion = Conv2dNormActivation(
            channels + channels,
            channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            activation_layer=nn.SiLU,
        )

        self.patch_size = patch_size
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)

        # Unfold (feature map -> patches)
        (patch_h, patch_w) = self.patch_size
        (B, C, H, W) = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h = new_h // patch_h  # n_h, n_w
        num_patch_w = new_w // patch_w
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)

        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate is True:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv_proj(x)
        x = self.conv_fusion(torch.concat((shortcut, x), dim=1))

        return x


# pylint: disable=invalid-name
class MobileViT_v1(BaseNet):
    default_size = (256, 256)

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
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        patch_size = (2, 2)
        attn_drop = 0.1
        depths = [2, 4, 3]
        strides = [1, 2, 1, 1]
        dims: list[int] = self.config["dims"]
        channels_a: list[int] = self.config["channels_a"]
        channels_b: list[int] = self.config["channels_b"]
        last_dim: int = self.config["last_dim"]
        expansion: int = self.config["expansion"]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            channels_a[0],
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=nn.SiLU,
            bias=True,
        )

        layers = []
        for i in range(1, len(channels_a)):
            if channels_a[i - 1] == channels_a[i] and strides[i - 1] == 1:
                shortcut = True
            else:
                shortcut = False

            layers.append(
                InvertedResidual(
                    channels_a[i - 1],
                    channels_a[i],
                    kernel_size=(3, 3),
                    stride=(strides[i - 1], strides[i - 1]),
                    padding=(1, 1),
                    expansion_factor=expansion,
                    shortcut=shortcut,
                    activation_layer=nn.SiLU,
                )
            )

        for idx, i in enumerate(range(0, len(channels_b), 2)):
            k = 2
            if i == 0:
                in_channels = channels_a[-1]
            else:
                in_channels = channels_b[i - 1]

            layers.append(
                InvertedResidual(
                    in_channels,
                    channels_b[i],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    expansion_factor=expansion,
                    shortcut=False,
                    activation_layer=nn.SiLU,
                )
            )
            layers.append(
                MobileVitBlock(
                    channels_b[i + 1],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    transformer_dim=dims[idx],
                    transformer_depth=depths[idx],
                    mlp_dim=dims[idx] * k,
                    patch_size=patch_size,
                    num_heads=4,
                    attn_drop=attn_drop,
                )
            )

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                channels_b[-2],
                last_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=nn.SiLU,
                bias=True,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = last_dim
        self.classifier = self.create_classifier()

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


registry.register_model_config(
    "mobilevit_v1_xxs",
    MobileViT_v1,
    config={
        "dims": [64, 80, 96],
        "channels_a": [16, 16, 24, 24, 24],
        "channels_b": [48, 48, 64, 64, 80, 80],
        "last_dim": 320,
        "expansion": 2,
    },
)
registry.register_model_config(
    "mobilevit_v1_xs",
    MobileViT_v1,
    config={
        "dims": [96, 120, 144],
        "channels_a": [16, 32, 48, 48, 48],
        "channels_b": [64, 64, 80, 80, 96, 96],
        "last_dim": 384,
        "expansion": 4,
    },
)
registry.register_model_config(
    "mobilevit_v1_s",
    MobileViT_v1,
    config={
        "dims": [144, 192, 240],
        "channels_a": [16, 32, 64, 64, 64],
        "channels_b": [96, 96, 128, 128, 160, 160],
        "last_dim": 640,
        "expansion": 4,
    },
)

registry.register_weights(
    "mobilevit_v1_xxs_il-common",
    {
        "description": "MobileViT v1 XXS model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 4.2,
                "sha256": "2b565a768ca21fd72d5ef5090ff0f8b725f3e1165cd8e56749815041e5254d26",
            }
        },
        "net": {"network": "mobilevit_v1_xxs", "tag": "il-common"},
    },
)
registry.register_weights(
    "mobilevit_v1_xs_il-common",
    {
        "description": "MobileViT v1 XS model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 8.1,
                "sha256": "193bcede7f0b9f4574673e95c23c6ca3b8eeb30254a32a85e93342f1d67db31b",
            }
        },
        "net": {"network": "mobilevit_v1_xs", "tag": "il-common"},
    },
)
