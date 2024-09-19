"""
DenseNet, adapted from
https://github.com/apache/mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/densenet.py
and
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

Paper "Densely Connected Convolutional Networks", https://arxiv.org/abs/1608.06993
"""

# Reference license: Apache-2.0 and BSD 3-Clause

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import BaseNet


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


class DenseNet(BaseNet):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        num_layers = int(self.net_param)

        if num_layers == 121:
            growth_rate = 32
            num_init_features = 64
            layer_list = [6, 12, 24, 16]

        elif num_layers == 161:
            growth_rate = 48
            num_init_features = 96
            layer_list = [6, 12, 36, 24]

        elif num_layers == 169:
            growth_rate = 32
            num_init_features = 64
            layer_list = [6, 12, 32, 32]

        elif num_layers == 201:
            growth_rate = 32
            num_init_features = 64
            layer_list = [6, 12, 48, 32]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

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
        layers = []
        num_features = num_init_features
        for i, num_layers in enumerate(layer_list):
            layers.append(DenseBlock(num_features, num_layers=num_layers, growth_rate=growth_rate))
            num_features = num_features + (num_layers * growth_rate)

            # Last block does not require transition
            if i != len(layer_list) - 1:
                layers.append(TransitionBlock(num_features, num_features // 2))
                num_features = num_features // 2

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)
