"""
Conv2Former, adapted from
https://github.com/HVision-NKU/Conv2Former/blob/main/models/conv2former.py

Paper "Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition", https://arxiv.org/abs/2211.11943
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.hornet import ChannelsFirstLayerNorm


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        expanded_channels = int(dim * mlp_ratio)
        self.norm = ChannelsFirstLayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, expanded_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.pos = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=expanded_channels,
        )
        self.fc2 = nn.Conv2d(expanded_channels, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        self.norm = ChannelsFirstLayerNorm(dim, eps=1e-6)
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                groups=dim,
            ),
        )
        self.v = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.proj = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.attn(x) * self.v(x)
        x = self.proj(x)

        return x


class Conv2FormerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int], mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.mlp = MLP(dim, mlp_ratio)

        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1 * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))

        return x


class Conv2FormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        kernel_size: tuple[int, int],
        mlp_ratio: float,
        drop_path: list[float],
        depth: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = nn.Sequential(
                ChannelsFirstLayerNorm(dim, eps=1e-6),
                nn.Conv2d(dim, out_dim, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        layers = []
        for i in range(depth):
            layers.append(
                Conv2FormerBlock(out_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio, drop_path=drop_path[i])
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class Conv2Former(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.blocks.(\d+)"

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

        mlp_ratios = [4.0, 4.0, 4.0, 4.0]
        kernel_size: tuple[int, int] = self.config["kernel_size"]
        dims: list[int] = self.config["dims"]
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)

        prev_dim = dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = Conv2FormerStage(
                prev_dim,
                dims[i],
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratios[i],
                drop_path=dpr[i],
                depth=depths[i],
                downsample=i > 0,
            )
            return_channels.append(dims[i])
            prev_dim = dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.Conv2d(dims[-1], 1280, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            ChannelsFirstLayerNorm(1280, eps=1e-6),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = 1280
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = dims[0]
        self.encoding_size = 1280

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        x = self.stem(x)
        x = mask_tensor(x, mask, patch_factor=self.max_stride // self.stem_stride, mask_token=mask_token)
        x = self.body(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            result["features"] = x
        if return_keys in ("all", "embedding"):
            result["embedding"] = self.features(x)

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "conv2former_n",
    Conv2Former,
    config={"kernel_size": (7, 7), "dims": [64, 128, 256, 512], "depths": [2, 2, 8, 2], "drop_path_rate": 0.1},
)
registry.register_model_config(
    "conv2former_t",
    Conv2Former,
    config={"kernel_size": (11, 11), "dims": [72, 144, 288, 576], "depths": [3, 3, 12, 3], "drop_path_rate": 0.15},
)
registry.register_model_config(
    "conv2former_s",
    Conv2Former,
    config={"kernel_size": (11, 11), "dims": [72, 144, 288, 576], "depths": [4, 4, 32, 4], "drop_path_rate": 0.2},
)
registry.register_model_config(
    "conv2former_b",
    Conv2Former,
    config={"kernel_size": (11, 11), "dims": [96, 192, 384, 768], "depths": [4, 4, 34, 4], "drop_path_rate": 0.2},
)
registry.register_model_config(
    "conv2former_l",
    Conv2Former,
    config={"kernel_size": (11, 11), "dims": [128, 256, 512, 1024], "depths": [4, 4, 48, 4], "drop_path_rate": 0.3},
)

registry.register_weights(
    "conv2former_n_il-common",
    {
        "description": "Conv2Former nano model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 54.3,
                "sha256": "47e93b34cfd7773f2a5c64bfe3c55a63e6c943dc1f4a130d5a4711ebdc96b632",
            }
        },
        "net": {"network": "conv2former_n", "tag": "il-common"},
    },
)
