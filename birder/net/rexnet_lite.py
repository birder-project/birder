"""
ReXNet Lite, adapted from
https://github.com/clovaai/rexnet/blob/master/rexnetv1_lite.py

Paper "Rethinking Channel Dimensions for Efficient Model Design",
https://arxiv.org/abs/2007.00992
"""

# Reference license: MIT

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
    initial_channels: int,
    final_channels: int,
    channel_divisor: int,
    kernel_config: str,
) -> list[tuple[int, int, int, int]]:
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    kernel_sizes = [int(element) for element in kernel_config]

    strides = sum([[stride] + [1] * (layers[idx] - 1) for idx, stride in enumerate(strides)], [])
    expansion_factors = [1] * layers[0] + [6] * sum(layers[1:])
    kernel_sizes = sum([[element] * layers[idx] for idx, element in enumerate(kernel_sizes)], [])

    depth = sum(layers)
    base_channels = initial_channels / width_multiplier if width_multiplier < 1.0 else initial_channels
    out_channels = []
    for block_idx in range(depth):
        base_channels_divisible = make_divisible(round(base_channels * width_multiplier), channel_divisor)
        if block_idx == 0:
            out_channels.append(base_channels_divisible)
        else:
            base_channels += final_channels / (depth - 1.0)
            base_channels_divisible = make_divisible(round(base_channels * width_multiplier), channel_divisor)
            out_channels.append(base_channels_divisible)

    return list(zip(out_channels, expansion_factors, kernel_sizes, strides))


class LinearBottleneckLite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        expansion_factor: int,
        kernel_size: tuple[int, int],
        channel_divisor: int,
    ) -> None:
        super().__init__()
        if stride == (1, 1) and in_channels <= out_channels:
            self.use_shortcut = True
        else:
            self.use_shortcut = False

        self.in_channels = in_channels

        layers = []
        if expansion_factor != 1:
            expanded_channels = make_divisible(in_channels * expansion_factor, channel_divisor)
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                    activation_layer=nn.ReLU6,
                )
            )
        else:
            expanded_channels = in_channels

        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=expanded_channels,
                bias=False,
                activation_layer=nn.ReLU6,
            )
        )
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


class ReXNet_Lite(DetectorBackbone):
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
        final_channels = 164
        width_multiplier: float = self.config["width_multiplier"]
        channel_divisor: int = self.config.get("channel_divisor", 8)
        dropout_rate: float = self.config.get("dropout_rate", 0.2)
        kernel_config: str = self.config.get("kernel_config", "333333")

        stem_base_channels = 32 / width_multiplier if width_multiplier < 1.0 else 32
        stem_channels = make_divisible(round(stem_base_channels * width_multiplier), channel_divisor)
        self.stem = Conv2dNormActivation(
            self.input_channels,
            stem_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
            activation_layer=nn.ReLU6,
        )

        net_settings = block_config(width_multiplier, initial_channels, final_channels, channel_divisor, kernel_config)

        layers: list[nn.Module] = []
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        stage_id = 0
        prev_channels = stem_channels
        for out_channels, expansion_factor, kernel_size, stride in net_settings:
            if stride > 1:
                stages[f"stage{stage_id}"] = nn.Sequential(*layers)
                return_channels.append(prev_channels)
                layers = []
                stage_id += 1

            layers.append(
                LinearBottleneckLite(
                    prev_channels,
                    out_channels,
                    stride=(stride, stride),
                    expansion_factor=expansion_factor,
                    kernel_size=(kernel_size, kernel_size),
                    channel_divisor=channel_divisor,
                )
            )
            prev_channels = out_channels

        stages[f"stage{stage_id}"] = nn.Sequential(*layers)
        return_channels.append(prev_channels)

        if width_multiplier > 1.0:
            penultimate_channels = int(1280 * width_multiplier)
        else:
            penultimate_channels = 1280

        head_channels = 1024
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                prev_channels,
                penultimate_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=nn.ReLU6,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(penultimate_channels, head_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        self.return_channels = return_channels[1:5]
        self.feature_dim = prev_channels
        self.embedding_size = head_channels
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

    def create_classifier(
        self, embed_dim: Optional[int] = None, head_bias: Optional[bool] = None, mlp_head: Optional[bool] = None
    ) -> nn.Module:
        assert head_bias is None, "head_bias customization is not supported"
        assert mlp_head is None, "mlp_head customization is not supported"

        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Sequential(
            nn.Conv2d(embed_dim, self.num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Flatten(1),
        )


registry.register_model_config("rexnet_lite_1_0", ReXNet_Lite, config={"width_multiplier": 1.0})
registry.register_model_config("rexnet_lite_1_3", ReXNet_Lite, config={"width_multiplier": 1.3})
registry.register_model_config("rexnet_lite_1_5", ReXNet_Lite, config={"width_multiplier": 1.5})
registry.register_model_config("rexnet_lite_2_0", ReXNet_Lite, config={"width_multiplier": 2.0})
