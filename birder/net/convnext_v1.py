"""
ConvNeXt v1, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py

Paper "A ConvNet for the 2020s", https://arxiv.org/abs/2201.03545
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.layers import LayerNorm2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        layer_scale: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding=(3, 3),
                groups=channels,
                bias=True,
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, 4 * channels),  # Same as 1x1 conv
            nn.GELU(),
            nn.Linear(4 * channels, channels),  # Same as 1x1 conv
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale, requires_grad=True)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.layer_scale * self.block(x)
        x = self.stochastic_depth(x)
        x += identity

        return x


# pylint: disable=invalid-name
class ConvNeXt_v1(DetectorBackbone):
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        layer_scale = 1e-6
        in_channels: list[int] = self.config["in_channels"]
        num_layers: list[int] = self.config["num_layers"]
        drop_path_rate: float = self.config["drop_path_rate"]
        out_channels = in_channels[1:] + [-1]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            in_channels[0],
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0),
            bias=True,
            norm_layer=LayerNorm2d,
            activation_layer=None,
        )

        total_stage_blocks = sum(num_layers)
        stage_block_id = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        layers = []
        for idx, (i, out, n) in enumerate(zip(in_channels, out_channels, num_layers)):
            # Bottlenecks
            for _ in range(n):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = drop_path_rate * stage_block_id / (total_stage_blocks - 1.0)
                layers.append(ConvNeXtBlock(i, layer_scale, sd_prob))
                stage_block_id += 1

            stages[f"stage{idx+1}"] = nn.Sequential(*layers)
            return_channels.append(i)
            layers = []

            # Down sampling
            if out != -1:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(i, eps=1e-6),
                        nn.Conv2d(i, out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                    )
                )

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(in_channels[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = in_channels[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "convnext_v1_tiny",
    ConvNeXt_v1,
    config={"in_channels": [96, 192, 384, 768], "num_layers": [3, 3, 9, 3], "drop_path_rate": 0.1},
)
registry.register_model_config(
    "convnext_v1_small",
    ConvNeXt_v1,
    config={"in_channels": [96, 192, 384, 768], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.4},
)
registry.register_model_config(
    "convnext_v1_base",
    ConvNeXt_v1,
    config={"in_channels": [128, 256, 512, 1024], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.5},
)
registry.register_model_config(
    "convnext_v1_large",
    ConvNeXt_v1,
    config={"in_channels": [192, 384, 768, 1536], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.5},
)
