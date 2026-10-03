"""
CPUBone, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cpubone.py

Paper "CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities",
https://arxiv.org/abs/2603.26425
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import stochastic_depth_rates


class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Module, shortcut: Optional[nn.Module], drop_path: float = 0.0) -> None:
        super().__init__()
        self.main = main
        self.shortcut = shortcut
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.main(x)

        return self.drop_path(self.main(x)) + self.shortcut(x)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        if kernel_size == 2:
            if stride == 1:
                # Keep the spatial resolution for the even-sized kernel
                self.conv = nn.Sequential(
                    nn.ZeroPad2d((1, 0, 1, 0)),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(0, 0),
                        groups=groups,
                        bias=bias,
                    ),
                )
            else:
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=(0, 0),
                    groups=groups,
                    bias=bias,
                )

        else:
            assert kernel_size % 2 == 1, "kernel size must be odd or equal to two"
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(kernel_size // 2, kernel_size // 2),
                groups=groups,
                bias=bias,
            )

        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class MBConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        mid_channels = 4 * channels

        self.inverted_conv = ConvLayer(
            channels,
            mid_channels,
            kernel_size=1,
            groups=2,
            bias=True,
            norm_layer=None,
            act_layer=nn.Hardswish,
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size=2,
            groups=mid_channels,
            bias=True,
            norm_layer=None,
            act_layer=nn.Hardswish,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            channels,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        spatial_bias: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=2,
            bias=spatial_bias,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)

        return x


class ConvAttention(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.num_heads = max(1, int((dim * 0.5) // 30))
        self.head_dim = int((dim // self.num_heads) * 0.5)
        self.scale = self.head_dim**-0.5
        self.stride = stride

        attention_dim = self.head_dim * self.num_heads
        self.conv_proj = ConvLayer(
            dim,
            dim,
            kernel_size=2,
            stride=stride,
            groups=dim,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
        )
        self.qkv = nn.Conv2d(dim, 3 * attention_dim, kernel_size=(1, 1), bias=False)
        self.out_proj = nn.Conv2d(attention_dim, dim, kernel_size=(1, 1))
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest") if stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        # A stride-two 2x2 projection rounds odd shapes down. Pad its bottom/right so
        # upsampling covers the residual feature map, then crop back to its exact shape.
        if self.stride > 1:
            pad_h = (-height) % self.stride
            pad_w = (-width) % self.stride
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.qkv(self.conv_proj(x))
        batch_size, _, proj_h, proj_w = x.size()
        x = x.reshape(batch_size, self.num_heads, 3 * self.head_dim, proj_h * proj_w)
        x = x.permute(0, 1, 3, 2)
        query, key, value = x.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(query, key, value, scale=self.scale)  # pylint: disable=not-callable
        x = x.permute(0, 1, 3, 2).reshape(batch_size, self.num_heads * self.head_dim, proj_h, proj_w)
        x = self.out_proj(x)
        x = self.upsample(x)

        return x[..., :height, :width]


class CPUBoneBlock(nn.Module):
    def __init__(self, channels: int, attention_stride: int, drop_path: float) -> None:
        super().__init__()
        attention = ConvAttention(channels, stride=attention_stride)
        context = ResidualBlock(
            nn.Sequential(nn.GroupNorm(num_groups=1, num_channels=channels), attention),
            nn.Identity(),
            drop_path,
        )
        context_mlp = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv2d(channels, 4 * channels, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(4 * channels, channels, kernel_size=(1, 1)),
            nn.Dropout(0.1),
        )
        context = nn.Sequential(context, ResidualBlock(context_mlp, nn.Identity(), drop_path))

        if channels < 256:
            local = FusedMBConv(
                channels,
                channels,
                kernel_size=2,
                stride=1,
                expand_ratio=4.0,
                spatial_bias=True,
            )
        else:
            local = MBConv(channels)

        self.block = nn.Sequential(context, ResidualBlock(local, nn.Identity(), drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CPUBone(DetectorBackbone):
    block_group_regex = r"body\.stage(\d+)\.(\d+)"

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

        widths: list[int] = self.config["widths"]
        depths: list[int] = self.config["depths"]
        head_widths: list[int] = self.config.get("head_widths", [1536, 1600])
        stem_expand_ratio: float = self.config.get("stem_expand_ratio", 2.0)
        downsample_expand_ratios: list[float] = self.config.get("downsample_expand_ratios", [4.0] * 4)
        drop_path_rate: float = self.config.get("drop_path_rate", 0.0)

        assert len(widths) == 5
        assert len(depths) == 5
        assert len(head_widths) == 2
        assert len(downsample_expand_ratios) == 4
        assert depths[1] > 0 and depths[2] > 0

        dpr = stochastic_depth_rates(drop_path_rate, sum(depths))
        block_idx = 0

        stem_layers: list[nn.Module] = [
            ConvLayer(
                self.input_channels,
                widths[0],
                kernel_size=3,
                stride=2,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.Hardswish,
            )
        ]
        for _ in range(depths[0]):
            block = FusedMBConv(
                widths[0],
                widths[0],
                kernel_size=3,
                stride=1,
                expand_ratio=stem_expand_ratio,
            )
            stem_layers.append(ResidualBlock(block, nn.Identity(), dpr[block_idx]))
            block_idx += 1

        self.stem = nn.Sequential(*stem_layers)

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        in_channels = widths[0]
        for stage_idx, (width, depth) in enumerate(zip(widths[1:], depths[1:], strict=True)):
            stage_num = stage_idx + 1
            stage_layers: list[nn.Module] = []
            if stage_num < 3:
                for layer_idx in range(depth):
                    stride = 2 if layer_idx == 0 else 1
                    block = FusedMBConv(
                        in_channels,
                        width,
                        kernel_size=3,
                        stride=stride,
                        expand_ratio=(downsample_expand_ratios[stage_idx] if stride == 2 else 4.0),
                    )
                    shortcut = None if stride == 2 else nn.Identity()
                    stage_layers.append(ResidualBlock(block, shortcut, dpr[block_idx]))
                    block_idx += 1
                    in_channels = width

            else:
                downsample = FusedMBConv(
                    in_channels,
                    width,
                    kernel_size=3,
                    stride=2,
                    expand_ratio=downsample_expand_ratios[stage_idx],
                )
                stage_layers.append(ResidualBlock(downsample, None))
                in_channels = width
                for _ in range(depth):
                    stage_layers.append(
                        CPUBoneBlock(
                            in_channels,
                            attention_stride=2 if stage_num == 3 else 1,
                            drop_path=dpr[block_idx],
                        )
                    )
                    block_idx += 1

            stages[f"stage{stage_num}"] = nn.Sequential(*stage_layers)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            ConvLayer(
                widths[-1],
                head_widths[0],
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.Hardswish,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Linear(head_widths[0], head_widths[1], bias=False),
            nn.LayerNorm(head_widths[1]),
            nn.Hardswish(),
        )
        self.return_channels = widths[1:]
        self.embedding_size = head_widths[1]
        self.classifier = self.create_classifier()

        self.max_stride = 32
        self.stem_stride = 2
        self.stem_width = widths[0]
        self.feature_dim = widths[-1]

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, stage in self.body.named_children():
            x = stage(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad_(False)

        for idx, stage in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in stage.parameters():
                param.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)


registry.register_model_config(
    "cpubone_n",
    CPUBone,
    config={"widths": [12, 24, 48, 96, 192], "depths": [0, 1, 1, 1, 2], "drop_path_rate": 0.05},
)
registry.register_model_config(
    "cpubone_t",
    CPUBone,
    config={"widths": [12, 24, 48, 96, 192], "depths": [0, 1, 1, 2, 3], "drop_path_rate": 0.05},
)
registry.register_model_config(
    "cpubone_s",
    CPUBone,
    config={"widths": [14, 28, 56, 112, 224], "depths": [0, 1, 1, 2, 3], "drop_path_rate": 0.05},
)
registry.register_model_config(
    "cpubone_b0",
    CPUBone,
    config={"widths": [16, 32, 64, 128, 256], "depths": [0, 1, 1, 3, 4], "drop_path_rate": 0.05},
)
registry.register_model_config(
    "cpubone_b1",
    CPUBone,
    config={
        "widths": [16, 32, 64, 128, 256],
        "depths": [0, 1, 1, 5, 5],
        "downsample_expand_ratios": [6.0, 6.0, 6.0, 6.0],
        "drop_path_rate": 0.05,
    },
)
registry.register_model_config(
    "cpubone_b2",
    CPUBone,
    config={
        "widths": [20, 40, 80, 160, 320],
        "depths": [0, 1, 1, 6, 6],
        "head_widths": [2304, 2560],
        "downsample_expand_ratios": [6.0, 6.0, 6.0, 6.0],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "cpubone_b3",
    CPUBone,
    config={
        "widths": [32, 64, 128, 256, 512],
        "depths": [1, 2, 3, 6, 6],
        "stem_expand_ratio": 4.0,
        "downsample_expand_ratios": [6.0, 6.0, 6.0, 6.0],
        "drop_path_rate": 0.1,
    },
)
