"""
RepGhost, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/repghost.py

Paper "RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization", https://arxiv.org/abs/2211.06088
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class RepGhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dw_size: int,
        stride: int,
        relu: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.reparameterized = reparameterized
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(kernel_size // 2, kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        if reparameterized is True:
            self.fusion_bn = None
        else:
            self.fusion_bn = nn.BatchNorm2d(out_channels)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(dw_size, dw_size),
                stride=(1, 1),
                padding=(dw_size // 2, dw_size // 2),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        if self.fusion_bn is not None:
            x2 = x2 + self.fusion_bn(x1)

        return self.relu(x2)

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        (kernel, bias) = self._get_kernel_bias()
        self.cheap_operation = nn.Conv2d(
            in_channels=self.cheap_operation[0].in_channels,
            out_channels=self.cheap_operation[0].out_channels,
            kernel_size=self.cheap_operation[0].kernel_size,
            padding=self.cheap_operation[0].padding,
            dilation=self.cheap_operation[0].dilation,
            groups=self.cheap_operation[0].groups,
            bias=True,
        )

        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias

        del self.fusion_bn
        self.fusion_bn = None
        self.reparameterized = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        (kernel, bias) = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        if self.fusion_bn is not None:
            (kernel1x1, bias_bn) = self._fuse_bn_tensor(nn.Identity(), self.fusion_bn, kernel.shape[0])
            kernel += F.pad(kernel1x1, [1, 1, 1, 1])
            bias += bias_bn

        return (kernel, bias)

    def _fuse_bn_tensor(
        self, conv: nn.Module, bn: nn.Module, in_channels: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if in_channels is None:
            in_channels = bn.running_mean.shape[0]

        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            device = bn.weight.device
            kernel = torch.ones(in_channels, 1, 1, 1, device=device)

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return (kernel * t, beta - running_mean * gamma / std)


class RepGhostBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dw_size: tuple[int, int],
        stride: tuple[int, int],
        se_ratio: float,
        reparameterized: bool,
    ) -> None:
        super().__init__()

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_channels, mid_channels, kernel_size=1, dw_size=3, stride=1, relu=True, reparameterized=reparameterized
        )

        # Depth-wise convolution
        if stride[0] > 1 or stride[1] > 1:
            self.conv_dw = nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=(dw_size[0] // 2, dw_size[1] // 2),
                groups=mid_channels,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)
        else:
            self.conv_dw = nn.Identity()
            self.bn_dw = nn.Identity()

        # Squeeze-and-excitation
        if se_ratio > 0:
            self.se = SqueezeExcitation(
                mid_channels, make_divisible(int(mid_channels * se_ratio), 4), activation=nn.Hardsigmoid
            )
        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_channels, out_channels, kernel_size=1, dw_size=3, stride=1, relu=False, reparameterized=reparameterized
        )

        # Shortcut
        if in_channels == out_channels and stride[0] == 1 and stride[1] == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=dw_size,
                    stride=stride,
                    padding=(dw_size[0] // 2, dw_size[1] // 2),
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.ghost1(x)
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.se(x)
        x = self.ghost2(x)
        x = x + self.shortcut(shortcut)

        return x


class RepGhost(DetectorBackbone):
    auto_register = True

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
        assert self.net_param is not None, "must set net-param"
        assert self.config is None, "config not supported"
        width = self.net_param

        self.reparameterized = False
        block_config: list[list[tuple[int, int, int, float, int]]] = [
            # kernel, expansion, channels, se ratio, stride
            # Stage 1
            [(3, 8, 16, 0, 1)],
            # Stage 2
            [(3, 24, 24, 0, 2)],
            [(3, 36, 24, 0, 1)],
            # Stage 3
            [(5, 36, 40, 0.25, 2)],
            [(5, 60, 40, 0.25, 1)],
            # Stage 4
            [(3, 120, 80, 0, 2)],
            [
                (3, 100, 80, 0, 1),
                (3, 120, 80, 0, 1),
                (3, 120, 80, 0, 1),
                (3, 240, 112, 0.25, 1),
                (3, 336, 112, 0.25, 1),
            ],
            # Stage 5
            [(5, 336, 160, 0.25, 2)],
            [
                (5, 480, 160, 0, 1),
                (5, 480, 160, 0.25, 1),
                (5, 480, 160, 0, 1),
                (5, 480, 160, 0.25, 1),
            ],
        ]

        stem_channels = make_divisible(16 * width, 4)
        self.stem = Conv2dNormActivation(
            self.input_channels, stem_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        i = 0
        layers: list[nn.Module] = []
        prev_channels = stem_channels
        for cfg in block_config:
            for k, exp_size, c, se_ratio, s in cfg:
                if s > 1:
                    stages[f"stage{i}"] = nn.Sequential(*layers)
                    return_channels.append(prev_channels)
                    layers = []
                    i += 1

                out_channels = make_divisible(c * width, 4)
                mid_channels = make_divisible(exp_size * width, 4)
                layers.append(
                    RepGhostBottleneck(
                        prev_channels,
                        mid_channels,
                        out_channels,
                        (k, k),
                        (s, s),
                        se_ratio=se_ratio,
                        reparameterized=self.reparameterized,
                    )
                )
                prev_channels = out_channels

        out_channels = make_divisible(exp_size * width * 2, 4)
        layers.append(
            Conv2dNormActivation(prev_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        stages[f"stage{i}"] = nn.Sequential(*layers)
        return_channels.append(out_channels)

        self.body = nn.Sequential(stages)
        prev_channels = out_channels
        out_channels = 1280
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(prev_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
        )
        self.return_channels = return_channels[1:5]
        self.embedding_size = out_channels
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

    def reparameterize_model(self) -> None:
        for module in self.modules():
            if hasattr(module, "reparameterize") is True:
                module.reparameterize()

        self.reparameterized = True


registry.register_weights(
    "repghost_1_il-common",
    {
        "description": "RepGhost 1.0x model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 12.8,
                "sha256": "37e211ec65c752ad79bbbaacea277f7d683d0b0f69d954a7ca7af46b9a1260e6",
            }
        },
        "net": {"network": "repghost", "net_param": 1, "tag": "il-common"},
    },
)
registry.register_weights(
    "repghost_1_il-common_reparameterized",
    {
        "description": "RepGhost 1.0x (reparameterized) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 12.6,
                "sha256": "e003e0498d63428305c10f879a0e2b999604795d417f07ea0da35ea925f794f5",
            }
        },
        "net": {"network": "repghost", "net_param": 1, "tag": "il-common_reparameterized", "reparameterized": True},
    },
)
