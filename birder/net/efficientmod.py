"""
EfficientMod, adapted from
https://github.com/ma-xu/EfficientMod/blob/main/models/EfficientMod.py

Paper "Efficient Modulation for Vision Networks", https://arxiv.org/abs/2403.19963
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.layers import LayerScale
from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, patch_stride: int, patch_pad: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_stride, patch_stride),
            padding=(patch_pad, patch_pad),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.num_heads = 8
        self.head_dim = max(dim // self.num_heads, 32)
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = nn.Linear(self.num_heads * self.head_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # pylint: disable=not-callable

        x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)

        return x


class EfficientModAttentionBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.layer_scale_1 = nn.Identity()
        self.drop_path_1 = StochasticDepth(drop_path, mode="row")

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=0.0)
        self.layer_scale_2 = nn.Identity()
        self.drop_path_2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = x + self.drop_path_1(self.layer_scale_1(self.attn(self.norm1(x))))
        x = x + self.drop_path_2(self.layer_scale_2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, C)

        return x


class ContextLayer(nn.Module):
    def __init__(self, in_dim: int, conv_dim: int, context_size: int) -> None:
        super().__init__()
        self.f = nn.Linear(in_dim, conv_dim)
        self.g = nn.Linear(conv_dim, in_dim)
        self.act = nn.GELU()
        self.context_layer = nn.Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=(context_size, context_size),
            stride=(1, 1),
            padding=(context_size // 2, context_size // 2),
            groups=conv_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.context_layer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.g(x)

        return x


class EfficientModMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, conv_group_dim: int, context_size: int) -> None:
        super().__init__()
        assert hidden_features % conv_group_dim == 0

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv_group_dim = conv_group_dim
        self.context_layer = ContextLayer(
            in_features,
            hidden_features // conv_group_dim,
            context_size=context_size,
        )
        self.expand_dim = hidden_features != in_features or conv_group_dim != 1
        self.act = nn.GELU() if self.expand_dim is True else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x = self.context_layer(x)

        x = self.fc1(x)
        x = self.act(x)

        if self.expand_dim is True:
            x = x * conv_x.repeat(1, 1, 1, self.conv_group_dim)
        else:
            x = x * conv_x

        x = self.fc2(x)

        return x


class EfficientModBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        conv_group_dim: int,
        context_size: int,
        layer_scale_init_value: Optional[float],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = EfficientModMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            conv_group_dim=conv_group_dim,
            context_size=context_size,
        )
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale(self.mlp(self.norm(x))))

        return x


class EfficientModStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        downsample: bool,
        depth: int,
        mlp_ratio: list[int],
        attention_mlp_ratio: float,
        layer_scale_init_value: Optional[float],
        drop_path: list[float],
        conv_group_dim: list[int],
        context_size: list[int],
        attention_depth: int,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = PatchEmbed(dim, out_dim, patch_size=3, patch_stride=2, patch_pad=1)
            dim = out_dim
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                EfficientModBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio[i],
                    conv_group_dim=conv_group_dim[i],
                    context_size=context_size[i],
                    layer_scale_init_value=layer_scale_init_value,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )
        for i in range(attention_depth):
            self.blocks.append(
                EfficientModAttentionBlock(
                    dim=dim,
                    mlp_ratio=attention_mlp_ratio,
                    drop_path=drop_path[depth + i],
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)

        return x


class EfficientMod(DetectorBackbone):
    block_group_regex = r"body\.stage(\d+)\.blocks\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]
        attention_depth: list[int] = self.config["attention_depth"]
        mlp_ratios: list[list[int]] = self.config["mlp_ratios"]
        attention_mlp_ratios: list[int] = self.config["attention_mlp_ratios"]
        drop_path_rate: float = self.config["drop_path_rate"]
        context_sizes: list[list[int]] = self.config["context_sizes"]
        conv_group_dims = mlp_ratios
        layer_scale_init_value = 1e-4

        num_stages = len(depths)
        assert len(embed_dims) == num_stages
        assert len(attention_depth) == num_stages
        assert len(mlp_ratios) == num_stages
        assert len(attention_mlp_ratios) == num_stages
        assert len(context_sizes) == num_stages

        self.stem = PatchEmbed(
            in_channels=self.input_channels, embed_dim=embed_dims[0], patch_size=7, patch_stride=4, patch_pad=3
        )

        block_depths = [depth + attention for depth, attention in zip(depths, attention_depth, strict=True)]
        dpr = [segment.tolist() for segment in torch.linspace(0, drop_path_rate, sum(block_depths)).split(block_depths)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        prev_dim = embed_dims[0]
        downsample = (False,) + (True,) * (len(depths) - 1)
        for stage_idx in range(len(depths)):
            stages[f"stage{stage_idx + 1}"] = EfficientModStage(
                dim=prev_dim,
                out_dim=embed_dims[stage_idx],
                downsample=downsample[stage_idx],
                depth=depths[stage_idx],
                mlp_ratio=mlp_ratios[stage_idx],
                attention_mlp_ratio=attention_mlp_ratios[stage_idx],
                layer_scale_init_value=layer_scale_init_value,
                drop_path=dpr[stage_idx],
                conv_group_dim=conv_group_dims[stage_idx],
                context_size=context_sizes[stage_idx],
                attention_depth=attention_depth[stage_idx],
            )
            prev_dim = embed_dims[stage_idx]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.LayerNorm(embed_dims[-1]),
            Permute([0, 3, 1, 2]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = embed_dims
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x.permute(0, 2, 3, 1))

        out = {}
        for name, stage in self.body.named_children():
            x = stage(x)
            if name in self.return_stages:
                out[name] = x.permute(0, 3, 1, 2).contiguous()

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x.permute(0, 2, 3, 1))
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.features(x)

        return x


registry.register_model_config(
    "efficientmod_xxs",
    EfficientMod,
    config={
        "embed_dims": [32, 64, 128, 256],
        "depths": [2, 2, 6, 2],
        "attention_depth": [0, 0, 1, 2],
        "attention_mlp_ratios": [0, 0, 4, 4],
        "mlp_ratios": [
            [1, 6, 1, 6],
            [1, 6, 1, 6],
            [1, 6, 1, 6, 1, 6],
            [1, 6, 1, 6],
        ],
        "context_sizes": [
            [7] * 10,
            [7] * 10,
            [7] * 20,
            [7] * 10,
        ],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "efficientmod_xs",
    EfficientMod,
    config={
        "embed_dims": [32, 64, 144, 288],
        "depths": [3, 3, 4, 2],
        "attention_depth": [0, 0, 3, 3],
        "attention_mlp_ratios": [4, 4, 4, 4],
        "mlp_ratios": [
            [1, 4, 1, 4] * 4,
            [1, 4, 1, 4] * 4,
            [1, 4, 1, 4] * 10,
            [1, 4, 1, 4] * 4,
        ],
        "context_sizes": [
            [7] * 10,
            [7] * 10,
            [7] * 20,
            [7] * 10,
        ],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "efficientmod_s",
    EfficientMod,
    config={
        "embed_dims": [32, 64, 144, 312],
        "depths": [4, 4, 8, 4],
        "attention_depth": [0, 0, 4, 4],
        "attention_mlp_ratios": [4, 4, 4, 5],
        "mlp_ratios": [
            [1, 6, 1, 6] * 4,
            [1, 6, 1, 6] * 4,
            [1, 6, 1, 6] * 10,
            [1, 6, 1, 6] * 4,
        ],
        "context_sizes": [
            [7] * 10,
            [7] * 10,
            [7] * 20,
            [7] * 10,
        ],
        "drop_path_rate": 0.02,
    },
)
registry.register_model_config(
    "efficientmod_s_conv",
    EfficientMod,
    config={
        "embed_dims": [40, 80, 160, 344],
        "depths": [4, 4, 12, 8],
        "attention_depth": [0, 0, 0, 0],
        "attention_mlp_ratios": [0, 0, 0, 0],
        "mlp_ratios": [
            [1, 6, 1, 6, 1, 6],
            [1, 6, 1, 6, 1, 6],
            [1, 6, 1, 6] * 5,
            [1, 6] * 8,
        ],
        "context_sizes": [
            [7] * 10,
            [7] * 10,
            [7] * 20,
            [7] * 12,
        ],
        "drop_path_rate": 0.02,
    },
)

registry.register_weights(
    "efficientmod_xxs_il-common",
    {
        "description": "EfficientMod XXS model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 17.5,
                "sha256": "8df1a2b6b24584f3f5e9e14533ef63ab92139799c662cf8eb6e28502e8316464",
            }
        },
        "net": {"network": "efficientmod_xxs", "tag": "il-common"},
    },
)
