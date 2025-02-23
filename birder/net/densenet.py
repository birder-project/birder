"""
DenseNet, adapted from
https://github.com/apache/mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/densenet.py
and
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

Paper "Densely Connected Convolutional Networks", https://arxiv.org/abs/1608.06993
"""

# Reference license: Apache-2.0 and BSD 3-Clause

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    Conv2dNormActivation(
                        in_channels + i * growth_rate,
                        4 * growth_rate,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        bias=False,
                    ),
                    nn.Conv2d(
                        4 * growth_rate,
                        growth_rate,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=False,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            dense_branch = layer(x)
            x = torch.concat((x, dense_branch), dim=1)

        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class DenseNet(DetectorBackbone):
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

        growth_rate: int = self.config["growth_rate"]
        num_init_features: int = self.config["num_init_features"]
        layer_list: list[int] = self.config["layer_list"]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                num_init_features,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # Add dense blocks
        num_features = num_init_features
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        layers = []
        for i, num_layers in enumerate(layer_list):

            layers.append(DenseBlock(num_features, num_layers=num_layers, growth_rate=growth_rate))
            num_features = num_features + (num_layers * growth_rate)

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(num_features)
            layers = []

            if i != len(layer_list) - 1:
                layers.append(TransitionBlock(num_features, num_features // 2))
                num_features = num_features // 2

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = num_features
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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias(
    "densenet_121", DenseNet, config={"growth_rate": 32, "num_init_features": 64, "layer_list": [6, 12, 24, 16]}
)
registry.register_alias(
    "densenet_161", DenseNet, config={"growth_rate": 48, "num_init_features": 96, "layer_list": [6, 12, 36, 24]}
)
registry.register_alias(
    "densenet_169", DenseNet, config={"growth_rate": 32, "num_init_features": 64, "layer_list": [6, 12, 32, 32]}
)
registry.register_alias(
    "densenet_201", DenseNet, config={"growth_rate": 32, "num_init_features": 64, "layer_list": [6, 12, 48, 32]}
)
