"""
Sequencer, adapted from
https://github.com/okojoalg/sequencer/blob/main/models/two_dim_sequencer.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/sequencer.py

Paper "Sequencer: Deep LSTM for Image Classification", https://arxiv.org/abs/2205.01972

Changes from original:
* Only supports the "cat" union (simplify the implementation)
"""

# Reference license: Apache-2.0 (both)

import math
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.base import staged_stochastic_depth_rates


class LSTM2d(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size

        self.fc = nn.Linear(2 * self.output_size, input_size)
        self.rnn_v = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bias=True,
            bidirectional=True,
        )
        self.rnn_h = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bias=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        v = x.permute(0, 2, 1, 3)
        v = v.reshape(-1, H, C)
        v, _ = self.rnn_v(v)
        v = v.reshape(B, W, H, -1)
        v = v.permute(0, 2, 1, 3)

        h = x.reshape(-1, W, C)
        h, _ = self.rnn_h(h)
        h = h.reshape(B, H, W, -1)

        x = torch.concat([v, h], dim=-1)
        x = self.fc(x)

        return x


class Sequencer2dBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_size: int,
        mlp_ratio: float,
        num_layers: int,
        drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.rnn = LSTM2d(dim, hidden_size, num_layers=num_layers)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, [int(mlp_ratio * dim), dim], activation_layer=nn.GELU, dropout=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.rnn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Downsample2d(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, patch_size: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(
            input_dim, output_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), padding=(0, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)

        return x


class Sequencer2dStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        patch_size: int,
        hidden_size: int,
        mlp_ratio: float,
        downsample: bool,
        num_layers: int,
        drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = Downsample2d(dim, dim_out, patch_size)
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            blocks.append(
                Sequencer2dBlock(
                    dim_out,
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    num_layers=num_layers,
                    drop=drop,
                    drop_path=drop_path[block_idx],
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class Sequencer2d(BaseNet):
    block_group_regex = r"body\.(\d+)\.blocks\.(\d+)"

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

        hidden_sizes = [48, 96, 96, 96]
        mlp_ratios = [3.0, 3.0, 3.0, 3.0]
        num_rnn_layers = 1
        drop_rate = 0.0
        layers: list[int] = self.config["layers"]
        patch_sizes: list[int] = self.config["patch_sizes"]
        embed_dims: list[int] = self.config["embed_dims"]
        drop_path_rate: float = self.config["drop_path_rate"]

        assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                embed_dims[0],
                kernel_size=(patch_sizes[0], patch_sizes[0]),
                stride=(patch_sizes[0], patch_sizes[0]),
                padding=(0, 0),
            ),
            Permute([0, 2, 3, 1]),
        )

        stages = []
        prev_dim = embed_dims[0]
        dpr = staged_stochastic_depth_rates(drop_path_rate, layers)
        for idx, embed_dim in enumerate(embed_dims):
            stages += [
                Sequencer2dStage(
                    prev_dim,
                    embed_dim,
                    depth=layers[idx],
                    patch_size=patch_sizes[idx],
                    hidden_size=hidden_sizes[idx],
                    mlp_ratio=mlp_ratios[idx],
                    downsample=idx > 0,
                    num_layers=num_rnn_layers,
                    drop=drop_rate,
                    drop_path=dpr[idx],
                )
            ]
            prev_dim = embed_dim

        self.body = nn.Sequential(*stages)
        self.features = nn.Sequential(
            nn.LayerNorm(prev_dim, eps=1e-6),
            Permute([0, 3, 1, 2]),  # B H W C -> B C H W)
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = prev_dim
        self.classifier = self.create_classifier()

        self.max_stride = math.prod(patch_sizes)
        self.stem_stride = patch_sizes[0]
        self.stem_width = embed_dims[0]
        self.feature_dim = prev_dim

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_in = m.weight.size(1) * math.prod(m.weight.shape[2:])
                truncated_normal_std = math.sqrt(
                    1.0 - 4.0 * math.exp(-2.0) / (math.sqrt(2.0 * math.pi) * math.erf(math.sqrt(2.0)))
                )
                std = math.sqrt(1.0 / fan_in) / truncated_normal_std
                nn.init.trunc_normal_(m.weight, std=std, a=-2.0 * std, b=2.0 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                std = 1.0 / math.sqrt(m.hidden_size)
                for parameter in m.parameters():
                    nn.init.uniform_(parameter, -std, std)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def flatten_features(self, features: torch.Tensor, include_special_tokens: bool = True) -> torch.Tensor:
        return features.flatten(1, 2)

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)


registry.register_model_config(
    "sequencer2d_s",
    Sequencer2d,
    config={
        "layers": [4, 3, 8, 3],
        "patch_sizes": [7, 2, 1, 1],
        "embed_dims": [192, 384, 384, 384],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "sequencer2d_m",
    Sequencer2d,
    config={
        "layers": [4, 3, 14, 3],
        "patch_sizes": [7, 2, 1, 1],
        "embed_dims": [192, 384, 384, 384],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "sequencer2d_l",
    Sequencer2d,
    config={
        "layers": [8, 8, 16, 4],
        "patch_sizes": [7, 2, 1, 1],
        "embed_dims": [192, 384, 384, 384],
        "drop_path_rate": 0.0,
    },
)
