"""
SqueezeNext 23v5 version, adapted from
https://github.com/amirgholami/SqueezeNext

Paper "SqueezeNext: Hardware-Aware Neural Network Design",  https://arxiv.org/abs/1803.10615
"""

# Reference license: BSD 2-Clause

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class SqnxtUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, project_identity: bool, vertical_first: bool
    ) -> None:
        super().__init__()
        if project_identity is True:
            self.identity = Conv2dNormActivation(
                in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0)
            )

        else:
            assert stride == 1
            assert in_channels == out_channels
            self.identity = nn.Identity()

        hidden_channels = out_channels // 2
        bottleneck_channels = out_channels // 4
        if vertical_first is True:
            spatial_kernels = ((3, 1), (1, 3))
            spatial_paddings = ((1, 0), (0, 1))
        else:
            spatial_kernels = ((1, 3), (3, 1))
            spatial_paddings = ((0, 1), (1, 0))

        self.block = nn.Sequential(
            Conv2dNormActivation(
                out_channels,
                hidden_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            Conv2dNormActivation(
                hidden_channels,
                bottleneck_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            Conv2dNormActivation(
                bottleneck_channels,
                hidden_channels,
                kernel_size=spatial_kernels[0],
                stride=(1, 1),
                padding=spatial_paddings[0],
            ),
            Conv2dNormActivation(
                hidden_channels,
                hidden_channels,
                kernel_size=spatial_kernels[1],
                stride=(1, 1),
                padding=spatial_paddings[1],
            ),
            Conv2dNormActivation(
                hidden_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.identity(x)
        identity = x
        x = self.block(x)
        x = x + identity
        x = self.relu(x)

        return x


class SqueezeNext(DetectorBackbone):
    default_size = (227, 227)

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

        width_scale: float = self.config["width_scale"]
        self.head_bias = self.config.get("head_bias", False)

        stem_width = 64
        embedding_size = 128
        channels_per_layers = [32, 64, 128, 256]
        layers_per_stage = [2, 4, 14, 1]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                stem_width,
                kernel_size=(5, 5),
                stride=(2, 2),
                padding=(0, 0),
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True),
        )

        in_channels = stem_width
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, lps in enumerate(layers_per_stage):
            layers = []
            for j in range(lps):
                if j == 0 and i != 0:
                    stride = 2
                else:
                    stride = 1

                out_channels = int(channels_per_layers[i] * width_scale)
                layers.append(
                    SqnxtUnit(
                        in_channels,
                        out_channels,
                        stride,
                        project_identity=j == 0,
                        vertical_first=j % 2 == 0,
                    )
                )
                in_channels = out_channels

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                embedding_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embedding_size
        self.classifier = self.create_classifier()

        self.max_stride = 32
        self.stem_stride = 4
        self.stem_width = stem_width
        self.feature_dim = in_channels

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
                bound = math.sqrt(3.0 / fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                bound = math.sqrt(3.0 / m.in_features)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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


registry.register_model_config("squeezenext_0_5", SqueezeNext, config={"width_scale": 0.5})
registry.register_model_config("squeezenext_1_0", SqueezeNext, config={"width_scale": 1.0})
registry.register_model_config("squeezenext_1_5", SqueezeNext, config={"width_scale": 1.5})
registry.register_model_config("squeezenext_2_0", SqueezeNext, config={"width_scale": 2.0})
