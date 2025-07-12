"""
ResNet v1, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Paper "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int], bottle_neck: bool, squeeze_excitation: bool
    ) -> None:
        super().__init__()
        if bottle_neck is True:
            self.block1 = nn.Sequential(
                Conv2dNormActivation(
                    in_channels,
                    out_channels // 4,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
                Conv2dNormActivation(
                    out_channels // 4,
                    out_channels // 4,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.Conv2d(
                    out_channels // 4,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.block1 = nn.Sequential(
                Conv2dNormActivation(
                    in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels == out_channels:
            self.block2 = nn.Identity()
        else:
            self.block2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        if squeeze_excitation is True:
            self.se = SqueezeExcitation(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block1(x)
        x = self.se(x)
        identity = self.block2(identity)
        x += identity
        x = self.relu(x)

        return x


# pylint: disable=invalid-name
class ResNet_v1(DetectorBackbone):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
        squeeze_excitation: bool = False,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        bottle_neck: bool = self.config["bottle_neck"]
        filter_list: list[int] = self.config["filter_list"]
        units: list[int] = self.config["units"]

        assert len(units) + 1 == len(filter_list)
        num_unit = len(units)

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                filter_list[0],
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # Generate body layers
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_unit):
            layers = []
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            layers.append(
                ResidualBlock(
                    filter_list[i],
                    filter_list[i + 1],
                    stride=stride,
                    bottle_neck=bottle_neck,
                    squeeze_excitation=squeeze_excitation,
                )
            )
            for _ in range(1, units[i]):
                layers.append(
                    ResidualBlock(
                        filter_list[i + 1],
                        filter_list[i + 1],
                        stride=(1, 1),
                        bottle_neck=bottle_neck,
                        squeeze_excitation=squeeze_excitation,
                    )
                )

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(filter_list[i + 1])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = filter_list[-1]
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "resnet_v1_18",
    ResNet_v1,
    config={"bottle_neck": False, "filter_list": [64, 64, 128, 256, 512], "units": [2, 2, 2, 2]},
)
registry.register_model_config(
    "resnet_v1_34",
    ResNet_v1,
    config={"bottle_neck": False, "filter_list": [64, 64, 128, 256, 512], "units": [3, 4, 6, 3]},
)
registry.register_model_config(
    "resnet_v1_50",
    ResNet_v1,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 4, 6, 3]},
)
registry.register_model_config(
    "resnet_v1_101",
    ResNet_v1,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 4, 23, 3]},
)
registry.register_model_config(
    "resnet_v1_152",
    ResNet_v1,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 8, 36, 3]},
)
registry.register_model_config(
    "resnet_v1_200",
    ResNet_v1,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 24, 36, 3]},
)
registry.register_model_config(
    "resnet_v1_269",
    ResNet_v1,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 30, 48, 8]},
)

registry.register_weights(
    "resnet_v1_50_arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/resnet_v1_50_arabian-peninsula/resolve/main",
        "description": "ResNet v1 50 model trained on the arabian-peninsula dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 95.8,
                "sha256": "fce0e7cd0d41ef546c3111801d1eeb8b821fabac88a6edc799144cf2637f9fd2",
            }
        },
        "net": {"network": "resnet_v1_50", "tag": "arabian-peninsula"},
    },
)
