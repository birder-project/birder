"""
EfficientNet v1, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

Paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
https://arxiv.org/abs/1905.11946
"""

# Reference license: BSD 3-Clause

import math
from collections import OrderedDict
from functools import partial
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


def adjust_channels(channels: int, width: float, min_value: Optional[int] = None) -> int:
    return make_divisible(channels * width, 8, min_value)


def adjust_depth(num_layers: int, depth: float) -> int:
    return int(math.ceil(num_layers * depth))


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        expand_ratio: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()

        if stride == (1, 1) and in_channels == out_channels:
            self.use_res_connect = True
        else:
            self.use_res_connect = False

        layers = []
        activation_layer = nn.SiLU

        # Expand
        expanded_channels = adjust_channels(in_channels, expand_ratio)
        if expanded_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise
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
                activation_layer=activation_layer,
            )
        )

        # Squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)
        layers.append(SqueezeExcitation(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # Project
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
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.block(x)
        if self.use_res_connect is True:
            return x + self.stochastic_depth(branch)

        return branch


# pylint: disable=invalid-name,too-many-locals
class EfficientNet_v1(DetectorBackbone):
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

        repeats = [1, 2, 2, 3, 3, 4, 1]
        in_channels = [32, 16, 24, 40, 80, 112, 192]
        out_channels = [16, 24, 40, 80, 112, 192, 320]
        kernel_size = [(3, 3), (3, 3), (5, 5), (3, 3), (5, 5), (5, 5), (3, 3)]
        expand_ratio = [1, 6, 6, 6, 6, 6, 6]
        strides = [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (2, 2), (1, 1)]
        width_coefficient: float = self.config["width_coefficient"]
        depth_coefficient: float = self.config["depth_coefficient"]
        dropout_rate: float = self.config["dropout_rate"]
        drop_path_rate: float = self.config.get("drop_path_rate", 0.2)

        self.dropout_rate = dropout_rate
        in_channels = [adjust_channels(ch, width_coefficient) for ch in in_channels]
        out_channels = [adjust_channels(ch, width_coefficient) for ch in out_channels]
        repeats = [adjust_depth(re, depth_coefficient) for re in repeats]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            in_channels[0],
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
            activation_layer=nn.SiLU,
        )

        layers: list[nn.Module] = []
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        stage_id = 0
        total_stage_blocks = sum(repeats)
        stage_block_id = 0
        for i, repeat in enumerate(repeats):
            for r in range(repeat):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = drop_path_rate * float(stage_block_id) / total_stage_blocks

                if r > 0:
                    in_ch = out_channels[i]
                    stride = (1, 1)

                else:
                    in_ch = in_channels[i]
                    stride = strides[i]

                if stride[0] > 1 or stride[1] > 1:
                    stages[f"stage{stage_id}"] = nn.Sequential(*layers)
                    return_channels.append(in_ch)
                    layers = []
                    stage_id += 1

                layers.append(
                    MBConv(
                        in_ch,
                        out_channels[i],
                        kernel_size=kernel_size[i],
                        stride=stride,
                        expand_ratio=expand_ratio[i],
                        stochastic_depth_prob=sd_prob,
                    )
                )

                stage_block_id += 1

        stages[f"stage{stage_id}"] = nn.Sequential(*layers)
        return_channels.append(out_channels[-1])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                out_channels[-1],
                out_channels[-1] * 4,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=nn.SiLU,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels[1:5]
        self.embedding_size = out_channels[-1] * 4
        self.classifier = self.create_classifier()

        # Weights initialization
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

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(embed_dim, self.num_classes),
        )


registry.register_alias(
    "efficientnet_v1_b0",
    EfficientNet_v1,  # input_resolution = 224
    config={"width_coefficient": 1.0, "depth_coefficient": 1.0, "dropout_rate": 0.2},
)
registry.register_alias(
    "efficientnet_v1_b1",
    EfficientNet_v1,  # input_resolution = 240
    config={"width_coefficient": 1.0, "depth_coefficient": 1.1, "dropout_rate": 0.2},
)
registry.register_alias(
    "efficientnet_v1_b2",
    EfficientNet_v1,  # input_resolution = 260
    config={"width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3},
)
registry.register_alias(
    "efficientnet_v1_b3",
    EfficientNet_v1,  # input_resolution = 300
    config={"width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3},
)
registry.register_alias(
    "efficientnet_v1_b4",
    EfficientNet_v1,  # input_resolution = 380
    config={"width_coefficient": 1.4, "depth_coefficient": 1.8, "dropout_rate": 0.4},
)
registry.register_alias(
    "efficientnet_v1_b5",
    EfficientNet_v1,  # input_resolution = 456
    config={"width_coefficient": 1.6, "depth_coefficient": 2.2, "dropout_rate": 0.4},
)
registry.register_alias(
    "efficientnet_v1_b6",
    EfficientNet_v1,  # input_resolution = 528
    config={"width_coefficient": 1.8, "depth_coefficient": 2.6, "dropout_rate": 0.5},
)
registry.register_alias(
    "efficientnet_v1_b7",
    EfficientNet_v1,  # input_resolution = 600
    config={"width_coefficient": 2.0, "depth_coefficient": 3.1, "dropout_rate": 0.5},
)

registry.register_weights(
    "efficientnet_v1_b0_il-common",
    {
        "description": "EfficientNet v1 B0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 17.4,
                "sha256": "9b8da7fa8095cb0bec88df85f56735d8c4a41e0f68cbf5dc610665fb30a5fbbc",
            }
        },
        "net": {"network": "efficientnet_v1_b0", "tag": "il-common"},
    },
)
registry.register_weights(
    "efficientnet_v1_b1_il-common",
    {
        "description": "EfficientNet v1 B1 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 27.1,
                "sha256": "e93b5f13cd93654fbec1455a3593a963aa415c21ad28c442583af57ac43e032f",
            }
        },
        "net": {"network": "efficientnet_v1_b1", "tag": "il-common"},
    },
)
registry.register_weights(
    "efficientnet_v1_b2_il-common",
    {
        "description": "EfficientNet v1 B2 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 31.8,
                "sha256": "3a62a94364c874e9d6b8e32aeb27470d88c6b3c15d690ee623e8a59697e4cba5",
            }
        },
        "net": {"network": "efficientnet_v1_b2", "tag": "il-common"},
    },
)
