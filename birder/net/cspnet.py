"""
CSPNet, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cspnet.py

Paper "CSPNet: A New Backbone that can Enhance Learning Capability of CNN", https://arxiv.org/abs/1911.11929

Changes from original:
* No partial shortcut (same as the TIMM implementation)
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottle_ratio: float,
        groups: int,
        squeeze_excitation: bool,
    ) -> None:
        super().__init__()
        mid_channels = int(round(out_channels * bottle_ratio))

        self.conv1 = Conv2dNormActivation(
            in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=nn.LeakyReLU
        )
        self.conv2 = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=groups,
            activation_layer=nn.LeakyReLU,
        )
        if squeeze_excitation is True:
            self.se = SqueezeExcitation(mid_channels, mid_channels // 16)
        else:
            self.se = nn.Identity()

        self.conv3 = Conv2dNormActivation(
            mid_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )
        self.act = nn.LeakyReLU()

        # Weights initialization
        nn.init.zeros_(self.conv3[1].weight)  # BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        x = x + shortcut
        x = self.act(x)

        return x


class DarkBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottle_ratio: float,
        groups: int,
        squeeze_excitation: bool,
    ) -> None:
        super().__init__()
        mid_channels = int(round(out_channels * bottle_ratio))

        self.conv1 = Conv2dNormActivation(
            in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=nn.LeakyReLU
        )
        if squeeze_excitation is True:
            self.se = SqueezeExcitation(mid_channels, mid_channels // 16)
        else:
            self.se = nn.Identity()

        self.conv2 = Conv2dNormActivation(
            mid_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=groups,
            activation_layer=nn.LeakyReLU,
        )

        # Weights initialization
        nn.init.zeros_(self.conv2[1].weight)  # BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.se(x)
        x = self.conv2(x)
        x = x + shortcut

        return x


class CrossStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
        block_ratio: float,
        bottle_ratio: float,
        expand_ratio: float,
        groups: int,
        down_growth: bool,
        cross_linear: bool,
        squeeze_excitation: bool,
        block_fn: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if down_growth is True:
            down_channels = out_channels
        else:
            down_channels = in_channels

        self.expand_channels = int(round(out_channels * expand_ratio))
        block_out_channels = int(round(out_channels * block_ratio))

        if stride != 1:
            self.conv_down = Conv2dNormActivation(
                in_channels,
                down_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
                activation_layer=nn.LeakyReLU,
            )
            prev_channels = down_channels
        else:
            self.conv_down = nn.Identity()
            prev_channels = in_channels

        if cross_linear is True:
            act = None
        else:
            act = nn.LeakyReLU

        self.conv_exp = Conv2dNormActivation(
            prev_channels,
            self.expand_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=act,
        )
        prev_channels = self.expand_channels // 2

        self.blocks = nn.Sequential()
        for _ in range(depth):
            self.blocks.append(
                block_fn(
                    in_channels=prev_channels,
                    out_channels=block_out_channels,
                    bottle_ratio=bottle_ratio,
                    groups=groups,
                    squeeze_excitation=squeeze_excitation,
                ),
            )
            prev_channels = block_out_channels

        self.conv_transition_b = Conv2dNormActivation(
            prev_channels,
            self.expand_channels // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.LeakyReLU,
        )
        self.conv_transition = Conv2dNormActivation(
            self.expand_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.LeakyReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = self.conv_exp(x)
        (xs, xb) = x.split(self.expand_channels // 2, dim=1)
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.concat([xs, xb], dim=1))

        return out


class CSPNet(DetectorBackbone):
    # pylint: disable=too-many-locals
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

        stem_channels: int = self.config["stem_channels"]
        stem_kernel: tuple[int, int] = self.config["stem_kernel"]
        stem_stride: tuple[int, int] = self.config["stem_stride"]
        stem_padding: tuple[int, int] = self.config["stem_padding"]
        stem_max_pool: bool = self.config["stem_max_pool"]
        depths: list[int] = self.config["depths"]
        filters: list[int] = self.config["filters"]
        strides: list[int] = self.config["strides"]
        block_ratio: float | list[float] = self.config["block_ratio"]
        bottle_ratio: float | list[float] = self.config["bottle_ratio"]
        expand_ratio: float | list[float] = self.config["expand_ratio"]
        groups: int = self.config["groups"]
        down_growth: bool = self.config["down_growth"]
        cross_linear: bool = self.config["cross_linear"]
        squeeze_excitation: bool = self.config["squeeze_excitation"]
        block_type_name: str = self.config["block_type_name"]

        if block_type_name == "BottleneckBlock":
            block_type = BottleneckBlock
        elif block_type_name == "DarkBlock":
            block_type = DarkBlock
        else:
            raise ValueError(f"Unknown block_type_name '{block_type_name}'")

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                stem_channels,
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=stem_padding,
                activation_layer=nn.LeakyReLU,
            )
        )
        if stem_max_pool is True:
            self.stem.append(nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        num_stages = len(depths)
        if not isinstance(block_ratio, Iterable):
            block_ratio = [block_ratio] * num_stages
        if not isinstance(bottle_ratio, Iterable):
            bottle_ratio = [bottle_ratio] * num_stages
        if not isinstance(expand_ratio, Iterable):
            expand_ratio = [expand_ratio] * num_stages

        prev_channels = stem_channels
        stage_idx = 5 - num_stages

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx in range(num_stages):
            stages[f"stage{stage_idx}"] = CrossStage(
                in_channels=prev_channels,
                out_channels=filters[idx],
                stride=strides[idx],
                depth=depths[idx],
                block_ratio=block_ratio[idx],
                bottle_ratio=bottle_ratio[idx],
                expand_ratio=expand_ratio[idx],
                groups=groups,
                down_growth=down_growth,
                cross_linear=cross_linear,
                squeeze_excitation=squeeze_excitation,
                block_fn=block_type,
            )
            return_channels.append(filters[idx])
            prev_channels = filters[idx]
            stage_idx += 1

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels[-4:]
        self.embedding_size = filters[-1]
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
    "csp_resnet_50",
    CSPNet,
    config={
        "stem_channels": 64,
        "stem_kernel": (7, 7),
        "stem_stride": (2, 2),
        "stem_padding": (3, 3),
        "stem_max_pool": True,
        "depths": [3, 3, 5, 2],
        "filters": [128, 256, 512, 1024],
        "strides": [1, 2, 2, 2],
        "block_ratio": 1.0,
        "bottle_ratio": 0.5,
        "expand_ratio": 2.0,
        "groups": 1,
        "down_growth": False,
        "cross_linear": True,
        "squeeze_excitation": False,
        "block_type_name": "BottleneckBlock",
    },
)
registry.register_model_config(
    "csp_resnext_50",
    CSPNet,
    config={
        "stem_channels": 64,
        "stem_kernel": (7, 7),
        "stem_stride": (2, 2),
        "stem_padding": (3, 3),
        "stem_max_pool": True,
        "depths": [3, 3, 5, 2],
        "filters": [256, 512, 1024, 2048],
        "strides": [1, 2, 2, 2],
        "block_ratio": 0.5,
        "bottle_ratio": 1.0,
        "expand_ratio": 1.0,
        "groups": 32,
        "down_growth": False,
        "cross_linear": True,
        "squeeze_excitation": False,
        "block_type_name": "BottleneckBlock",
    },
)
registry.register_model_config(
    "csp_darknet_53",
    CSPNet,
    config={
        "stem_channels": 32,
        "stem_kernel": (3, 3),
        "stem_stride": (1, 1),
        "stem_padding": (1, 1),
        "stem_max_pool": False,
        "depths": [1, 2, 8, 8, 4],
        "filters": [64, 128, 256, 512, 1024],
        "strides": [2, 2, 2, 2, 2],
        "block_ratio": [1.0, 0.5, 0.5, 0.5, 0.5],
        "bottle_ratio": [0.5, 1.0, 1.0, 1.0, 1.0],
        "expand_ratio": [2.0, 1.0, 1.0, 1.0, 1.0],
        "groups": 1,
        "down_growth": True,
        "cross_linear": False,
        "squeeze_excitation": False,
        "block_type_name": "DarkBlock",
    },
)
registry.register_model_config(
    "csp_se_resnet_50",
    CSPNet,
    config={
        "stem_channels": 64,
        "stem_kernel": (7, 7),
        "stem_stride": (2, 2),
        "stem_padding": (3, 3),
        "stem_max_pool": True,
        "depths": [3, 3, 5, 2],
        "filters": [128, 256, 512, 1024],
        "strides": [1, 2, 2, 2],
        "block_ratio": 1.0,
        "bottle_ratio": 0.5,
        "expand_ratio": 2.0,
        "groups": 1,
        "down_growth": False,
        "cross_linear": True,
        "squeeze_excitation": True,
        "block_type_name": "BottleneckBlock",
    },
)
registry.register_model_config(
    "csp_se_resnext_50",
    CSPNet,
    config={
        "stem_channels": 64,
        "stem_kernel": (7, 7),
        "stem_stride": (2, 2),
        "stem_padding": (3, 3),
        "stem_max_pool": True,
        "depths": [3, 3, 5, 2],
        "filters": [256, 512, 1024, 2048],
        "strides": [1, 2, 2, 2],
        "block_ratio": 0.5,
        "bottle_ratio": 1.0,
        "expand_ratio": 1.0,
        "groups": 32,
        "down_growth": False,
        "cross_linear": True,
        "squeeze_excitation": True,
        "block_type_name": "BottleneckBlock",
    },
)
registry.register_model_config(
    "csp_se_darknet_53",
    CSPNet,
    config={
        "stem_channels": 32,
        "stem_kernel": (3, 3),
        "stem_stride": (1, 1),
        "stem_padding": (1, 1),
        "stem_max_pool": False,
        "depths": [1, 2, 8, 8, 4],
        "filters": [64, 128, 256, 512, 1024],
        "strides": [2, 2, 2, 2, 2],
        "block_ratio": [1.0, 0.5, 0.5, 0.5, 0.5],
        "bottle_ratio": [0.5, 1.0, 1.0, 1.0, 1.0],
        "expand_ratio": [2.0, 1.0, 1.0, 1.0, 1.0],
        "groups": 1,
        "down_growth": True,
        "cross_linear": False,
        "squeeze_excitation": True,
        "block_type_name": "DarkBlock",
    },
)
