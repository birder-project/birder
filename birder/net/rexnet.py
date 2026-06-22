"""
ReXNet, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/rexnet.py

Paper "Rethinking Channel Dimensions for Efficient Model Design",
https://arxiv.org/abs/2007.00992
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


def block_config(
    width_multiplier: float,
    depth_multiplier: float,
    initial_channels: int,
    final_channels: int,
    se_ratio: float,
    channel_divisor: int,
) -> list[tuple[int, float, int, float]]:
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]

    layers = [math.ceil(layer * depth_multiplier) for layer in layers]
    strides = sum([[stride] + [1] * (layers[idx] - 1) for idx, stride in enumerate(strides)], [])
    expansion_factors = [1] * layers[0] + [6] * sum(layers[1:])

    depth = sum(layers)
    base_channels = initial_channels / width_multiplier if width_multiplier < 1.0 else initial_channels
    out_channels = []
    for _ in range(depth):
        out_channels.append(make_divisible(round(base_channels * width_multiplier), channel_divisor))
        base_channels += final_channels / depth

    se_ratios = [0.0] * (layers[0] + layers[1]) + [se_ratio] * sum(layers[2:])

    return list(zip(out_channels, expansion_factors, strides, se_ratios))


class ReXNetSqueezeExcitation(nn.Module):
    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn = nn.BatchNorm2d(squeeze_channels)
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.bn(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)

        return x * scale


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        expansion_factor: float,
        se_ratio: float,
        channel_divisor: int,
    ) -> None:
        super().__init__()
        if stride == (1, 1) and in_channels <= out_channels:
            self.use_shortcut = True
        else:
            self.use_shortcut = False

        self.in_channels = in_channels

        layers = []
        if expansion_factor != 1.0:
            expanded_channels = make_divisible(round(in_channels * expansion_factor), channel_divisor)
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                    activation_layer=nn.SiLU,
                )
            )
        else:
            expanded_channels = in_channels

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=expanded_channels,
                bias=False,
                activation_layer=None,
            )
        )
        if se_ratio > 0:
            squeeze_channels = make_divisible(int(expanded_channels * se_ratio), channel_divisor)
            layers.append(ReXNetSqueezeExcitation(expanded_channels, squeeze_channels))

        layers.append(nn.ReLU6(inplace=True))
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_shortcut is True:
            out = torch.concat([out[:, 0 : self.in_channels] + x, out[:, self.in_channels :]], dim=1)

        return out


class ReXNet(DetectorBackbone):
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

        initial_channels = 16
        final_channels = 180
        width_multiplier: float = self.config["width_multiplier"]
        depth_multiplier: float = self.config.get("depth_multiplier", 1.0)
        se_ratio: float = self.config.get("se_ratio", 1.0 / 12.0)
        channel_divisor: int = self.config.get("channel_divisor", 1)
        dropout_rate: float = self.config.get("dropout_rate", 0.2)

        stem_base_channels = 32 / width_multiplier if width_multiplier < 1.0 else 32
        stem_channels = make_divisible(round(stem_base_channels * width_multiplier), channel_divisor)
        self.stem = Conv2dNormActivation(
            self.input_channels,
            stem_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
            activation_layer=nn.SiLU,
        )

        net_settings = block_config(
            width_multiplier, depth_multiplier, initial_channels, final_channels, se_ratio, channel_divisor
        )

        layers: list[nn.Module] = []
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        stage_id = 0
        prev_channels = stem_channels
        for out_channels, expansion_factor, stride, block_se_ratio in net_settings:
            if stride > 1:
                stages[f"stage{stage_id}"] = nn.Sequential(*layers)
                return_channels.append(prev_channels)
                layers = []
                stage_id += 1

            layers.append(
                LinearBottleneck(
                    prev_channels,
                    out_channels,
                    stride=(stride, stride),
                    expansion_factor=expansion_factor,
                    se_ratio=block_se_ratio,
                    channel_divisor=channel_divisor,
                )
            )
            prev_channels = out_channels

        stages[f"stage{stage_id}"] = nn.Sequential(*layers)
        return_channels.append(prev_channels)

        penultimate_channels = make_divisible(1280 * width_multiplier, channel_divisor)
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                prev_channels,
                penultimate_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=nn.SiLU,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=dropout_rate, inplace=True),
        )
        self.return_channels = return_channels[1:5]
        self.feature_dim = prev_channels
        self.embedding_size = penultimate_channels
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
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
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)


registry.register_model_config("rexnet_1_0", ReXNet, config={"width_multiplier": 1.0})
registry.register_model_config("rexnet_1_3", ReXNet, config={"width_multiplier": 1.3})
registry.register_model_config("rexnet_1_5", ReXNet, config={"width_multiplier": 1.5})
registry.register_model_config("rexnet_2_0", ReXNet, config={"width_multiplier": 2.0})
registry.register_model_config("rexnet_3_0", ReXNet, config={"width_multiplier": 3.0})
