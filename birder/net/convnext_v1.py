"""
ConvNeXt v1, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py

Paper "A ConvNet for the 2020s", https://arxiv.org/abs/2201.03545
"""

# Reference license: BSD 3-Clause

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)
        x = x.permute(0, 3, 1, 2)

        return x


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
class ConvNeXt_v1(BaseNet):
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
        net_param = int(self.net_param)

        layer_scale = 1e-6

        # 0 = tiny
        # 1 = small
        # 2 = base
        # 3 = large
        if net_param == 0:
            in_channels = [96, 192, 384, 768]
            out_channels = [192, 384, 768, -1]
            num_layers = [3, 3, 9, 3]
            stochastic_depth_prob = 0.1

        elif net_param == 1:
            in_channels = [96, 192, 384, 768]
            out_channels = [192, 384, 768, -1]
            num_layers = [3, 3, 27, 3]
            stochastic_depth_prob = 0.4

        elif net_param == 2:
            in_channels = [128, 256, 512, 1024]
            out_channels = [256, 512, 1024, -1]
            num_layers = [3, 3, 27, 3]
            stochastic_depth_prob = 0.5

        elif net_param == 3:
            in_channels = [192, 384, 768, 1536]
            out_channels = [384, 768, 1536, -1]
            num_layers = [3, 3, 27, 3]
            stochastic_depth_prob = 0.5

        else:
            raise ValueError(f"net_param = {net_param} not supported")

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

        layers = []
        total_stage_blocks = sum(num_layers)
        stage_block_id = 0
        for i, out, n in zip(in_channels, out_channels, num_layers):
            # Bottlenecks
            stage = []
            for _ in range(n):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(ConvNeXtBlock(i, layer_scale, sd_prob))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

            # Down sampling
            if out != -1:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(i, eps=1e-6),
                        nn.Conv2d(i, out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                    )
                )

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(in_channels[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.embedding_size = in_channels[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias("convnext_v1_tiny", ConvNeXt_v1, 0)
registry.register_alias("convnext_v1_small", ConvNeXt_v1, 1)
registry.register_alias("convnext_v1_base", ConvNeXt_v1, 2)
registry.register_alias("convnext_v1_large", ConvNeXt_v1, 3)
