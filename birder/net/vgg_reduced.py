"""
A modern approach to VGG
"""

from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import DetectorBackbone


# pylint: disable=invalid-name
class Vgg_Reduced(DetectorBackbone):
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

        filters = [64, 128, 256, 512, 512]
        if num_layers == 11:
            repeats = [1, 1, 2, 2, 2]

        elif num_layers == 13:
            repeats = [2, 2, 2, 2, 2]

        elif num_layers == 16:
            repeats = [2, 2, 3, 3, 3]

        elif num_layers == 19:
            repeats = [2, 2, 4, 4, 4]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, num in enumerate(repeats):
            layers = []
            for j in range(num):
                if i == 0 and j == 0:
                    in_channels = self.input_channels
                elif j == 0:
                    in_channels = filters[i - 1]
                else:
                    in_channels = filters[i]

                layers.append(
                    Conv2dNormActivation(
                        in_channels,
                        filters[i],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True,
                    )
                )

            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
            stages[f"stage{i}"] = nn.Sequential(*layers)
            return_channels.append(filters[i])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(filters[-1], 1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = 1024
        self.return_channels = return_channels[1:5]
        self.classifier = self.create_classifier()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.features(x)
