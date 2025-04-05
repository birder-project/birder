"""
Paper "YOLOv3: An Incremental Improvement", https://arxiv.org/abs/1804.02767
"""

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        mid_channels = channels // 2
        self.block = nn.Sequential(
            Conv2dNormActivation(
                channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=nn.LeakyReLU
            ),
            Conv2dNormActivation(
                mid_channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation_layer=nn.LeakyReLU
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x)
        x = x + identity

        return x


class Darknet(DetectorBackbone):
    default_size = (256, 256)

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

        filters: list[int] = self.config.get("filters", [64, 128, 256, 512, 1024])
        repeats: list[int] = self.config["repeats"]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation_layer=nn.LeakyReLU,
            ),
            Conv2dNormActivation(
                32,
                filters[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.LeakyReLU,
            ),
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, num in enumerate(repeats):
            layers = []
            if i > 0:
                layers.append(
                    Conv2dNormActivation(
                        filters[i - 1],
                        filters[i],
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        activation_layer=nn.LeakyReLU,
                    )
                )

            for _ in range(num):
                layers.append(ResidualBlock(filters[i]))

            stages[f"stage{i}"] = nn.Sequential(*layers)
            return_channels.append(filters[i])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = filters[-1]
        self.return_channels = return_channels[1:5]
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


registry.register_alias("darknet_17", Darknet, config={"repeats": [1, 1, 1, 1, 1]})
registry.register_alias("darknet_21", Darknet, config={"repeats": [1, 1, 1, 2, 2]})
registry.register_alias("darknet_53", Darknet, config={"repeats": [1, 2, 8, 8, 4]})
