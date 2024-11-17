"""
MobileNet v4, adapted from
https://github.com/jaiwei98/MobileNetV4-pytorch/blob/main/mobilenet/mobilenetv4.py

Paper "MobileNetV4 -- Universal Models for the Mobile Ecosystem", https://arxiv.org/abs/2404.10518
"""

# Reference license: MIT

from collections import OrderedDict
from collections.abc import Callable
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


class ConvNormActConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        activation: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.activation = activation


class InvertedResidualConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        width_multi: float,
        use_se: bool,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        self.in_channels = make_divisible(in_channels, 8)
        self.out_channels = make_divisible(out_channels, 8)
        self.expanded_channels = make_divisible(in_channels * width_multi, 8)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        self.activation = activation


class UniversalInvertedBottleneckConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        start_dw_kernel_size: Optional[tuple[int, int]],
        middle_dw_kernel_size: Optional[tuple[int, int]],
        stride: tuple[int, int],
        middle_dw_downsample: bool,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.stride = stride
        self.middle_dw_downsample = middle_dw_downsample
        self.activation = activation

        if stride[0] == 1 and stride[1] == 1 and in_channels == out_channels:
            self.shortcut = True
        else:
            self.shortcut = False


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones([dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        if self.inplace is True:
            return x.mul_(gamma)

        return x * gamma


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, stochastic_depth_prob: float) -> None:
        super().__init__()
        if cnf.stride[0] == 1 and cnf.stride[1] == 1 and cnf.in_channels == cnf.out_channels:
            self.shortcut = True
        else:
            self.shortcut = False

        layers = []
        # Expand
        if cnf.expanded_channels != cnf.in_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.in_channels,
                    cnf.expanded_channels,
                    kernel_size=cnf.kernel_size,
                    stride=cnf.stride,
                    padding=cnf.padding,
                    activation_layer=cnf.activation,
                    inplace=None,
                    bias=False,
                )
            )

        if cnf.use_se is True:
            # Depthwise
            layers.append(
                Conv2dNormActivation(
                    cnf.expanded_channels,
                    cnf.expanded_channels,
                    kernel_size=cnf.kernel_size,
                    stride=cnf.stride,
                    padding=cnf.padding,
                    groups=cnf.expanded_channels,
                    activation_layer=cnf.activation,
                    inplace=None,
                    bias=False,
                )
            )
            squeeze_channels = make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(cnf.expanded_channels, squeeze_channels, scale_activation=nn.Hardsigmoid))

        # Project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
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
        if self.shortcut is True:
            x = self.stochastic_depth(x)
            return x + self.block(x)

        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(
        self, cnf: UniversalInvertedBottleneckConfig, layer_scale_init_value: float, stochastic_depth_prob: float
    ) -> None:
        super().__init__()
        self.shortcut = cnf.shortcut

        if cnf.start_dw_kernel_size is not None:
            if cnf.middle_dw_downsample is True:
                s = (1, 1)
            else:
                s = cnf.stride

            self.start_dw_conv = Conv2dNormActivation(
                cnf.in_channels,
                cnf.in_channels,
                kernel_size=cnf.start_dw_kernel_size,
                stride=s,
                padding=((cnf.start_dw_kernel_size[0] - 1) // 2, (cnf.start_dw_kernel_size[1] - 1) // 2),
                groups=cnf.in_channels,
                activation_layer=None,
                bias=False,
            )

        else:
            self.start_dw_conv = nn.Identity()

        expand_channels = make_divisible(cnf.in_channels * cnf.expand_ratio, 8)
        self.expand_conv = Conv2dNormActivation(
            cnf.in_channels,
            expand_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=cnf.activation,
            inplace=None,
            bias=False,
        )

        if cnf.middle_dw_kernel_size is not None:
            if cnf.middle_dw_downsample is True:
                s = cnf.stride
            else:
                s = (1, 1)

            self.middle_dw_conv = Conv2dNormActivation(
                expand_channels,
                expand_channels,
                kernel_size=cnf.middle_dw_kernel_size,
                stride=s,
                padding=((cnf.middle_dw_kernel_size[0] - 1) // 2, (cnf.middle_dw_kernel_size[1] - 1) // 2),
                groups=expand_channels,
                activation_layer=cnf.activation,
                inplace=None,
                bias=False,
            )

        else:
            self.middle_dw_conv = nn.Identity()

        self.proj_conv = Conv2dNormActivation(
            expand_channels,
            cnf.out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=None,
            bias=False,
        )

        if layer_scale_init_value > 0:
            self.layer_scale = LayerScale2d(cnf.out_channels, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.start_dw_conv(x)
        x = self.expand_conv(x)
        x = self.middle_dw_conv(x)
        x = self.proj_conv(x)
        x = self.layer_scale(x)
        if self.shortcut is True:
            x = self.stochastic_depth(x)
            return x + shortcut

        return x


# pylint: disable=invalid-name
class MobileNet_v4(DetectorBackbone):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        dropout: float = self.config["dropout"]
        stochastic_depth_prob: float = self.config["stochastic_depth_prob"]
        stem_settings: ConvNormActConfig = self.config["stem_settings"]
        net_settings: list[Any] = self.config["net_settings"]
        last_stage_settings: list[ConvNormActConfig] = self.config["last_stage_settings"]
        features_stage_settings: ConvNormActConfig = self.config["features_stage_settings"]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            stem_settings.out_channels,
            kernel_size=stem_settings.kernel,
            stride=stem_settings.stride,
            padding=stem_settings.padding,
            activation_layer=stem_settings.activation,
        )

        layers: list[nn.Module] = []
        total_stage_blocks = len(net_settings) + len(last_stage_settings)
        i = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx, block_settings in enumerate(net_settings):
            # Adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * float(idx) / total_stage_blocks

            if idx > 0 and (block_settings.stride[0] > 1 or block_settings.stride[1] > 1):
                stages[f"stage{i+1}"] = nn.Sequential(*layers)
                return_channels.append(net_settings[idx - 1].out_channels)
                layers = []
                i += 1

            if isinstance(block_settings, ConvNormActConfig):
                layers.append(
                    Conv2dNormActivation(
                        block_settings.in_channels,
                        block_settings.out_channels,
                        kernel_size=block_settings.kernel,
                        stride=block_settings.stride,
                        padding=block_settings.padding,
                        activation_layer=block_settings.activation,
                    )
                )

            elif isinstance(block_settings, InvertedResidualConfig):
                layers.append(InvertedResidual(block_settings, sd_prob))

            elif isinstance(block_settings, UniversalInvertedBottleneckConfig):
                layers.append(UniversalInvertedBottleneck(block_settings, 0.0, sd_prob))

            else:
                raise ValueError("Unknown config")

        stages[f"stage{i+1}"] = nn.Sequential(*layers)
        return_channels.append(net_settings[-1].out_channels)
        layers = []
        i += 1
        for block_settings in last_stage_settings:
            layers.append(
                Conv2dNormActivation(
                    block_settings.in_channels,
                    block_settings.out_channels,
                    kernel_size=block_settings.kernel,
                    stride=block_settings.stride,
                    padding=block_settings.padding,
                    activation_layer=block_settings.activation,
                )
            )
        stages[f"stage{i+1}"] = nn.Sequential(*layers)
        return_channels.append(last_stage_settings[-1].out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Conv2dNormActivation(
                features_stage_settings.in_channels,
                features_stage_settings.out_channels,
                kernel_size=features_stage_settings.kernel,
                stride=features_stage_settings.stride,
                padding=features_stage_settings.padding,
                activation_layer=features_stage_settings.activation,
            ),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
        )
        self.return_channels = return_channels[:4]
        self.embedding_size = features_stage_settings.out_channels
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def transform_to_backbone(self) -> None:
        self.body.stage5 = nn.Identity()
        self.features = nn.Identity()
        self.classifier = nn.Identity()

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
    "mobilenet_v4_s",
    MobileNet_v4,
    config={
        "dropout": 0.3,
        "stochastic_depth_prob": 0.0,
        "stem_settings": ConvNormActConfig(0, 32, (3, 3), (2, 2), (1, 1)),
        "net_settings": [
            # Stage 1
            ConvNormActConfig(32, 32, (3, 3), (2, 2), (1, 1)),
            ConvNormActConfig(32, 32, (1, 1), (1, 1), (0, 0)),
            # Stage 2
            ConvNormActConfig(32, 96, (3, 3), (2, 2), (1, 1)),
            ConvNormActConfig(96, 64, (1, 1), (1, 1), (0, 0)),
            # Stage 3
            UniversalInvertedBottleneckConfig(64, 96, 3.0, (5, 5), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(96, 96, 2.0, None, (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(96, 96, 2.0, None, (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(96, 96, 2.0, None, (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(96, 96, 2.0, None, (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(96, 96, 4.0, (3, 3), None, (1, 1), True),
            # Stage 4
            UniversalInvertedBottleneckConfig(96, 128, 6.0, (3, 3), (3, 3), (2, 2), True),
            UniversalInvertedBottleneckConfig(128, 128, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(128, 128, 4.0, None, (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(128, 128, 3.0, None, (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(128, 128, 4.0, None, (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(128, 128, 4.0, None, (3, 3), (1, 1), True),
        ],
        "last_stage_settings": [
            ConvNormActConfig(128, 960, (1, 1), (1, 1), (0, 0)),
        ],
        "features_stage_settings": ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0)),
    },
)
registry.register_alias(
    "mobilenet_v4_m",
    MobileNet_v4,
    config={
        "dropout": 0.2,
        "stochastic_depth_prob": 0.075,
        "stem_settings": ConvNormActConfig(0, 32, (3, 3), (2, 2), (1, 1)),
        "net_settings": [
            # Stage 1
            InvertedResidualConfig(32, 48, (3, 3), (2, 2), (1, 1), 4.0, False),
            # Stage 2
            UniversalInvertedBottleneckConfig(48, 80, 4.0, (3, 3), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(80, 80, 2.0, (3, 3), (3, 3), (1, 1), True),
            # Stage 3
            UniversalInvertedBottleneckConfig(80, 160, 6.0, (3, 3), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(160, 160, 2.0, None, None, (1, 1), True),
            UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), None, (1, 1), True),
            # Stage 4
            UniversalInvertedBottleneckConfig(160, 256, 6.0, (5, 5), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, (3, 3), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, (3, 3), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, None, None, (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, (3, 3), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 2.0, (3, 3), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, None, None, (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 4.0, None, None, (1, 1), True),
            UniversalInvertedBottleneckConfig(256, 256, 2.0, (5, 5), None, (1, 1), True),
        ],
        "last_stage_settings": [
            ConvNormActConfig(256, 960, (1, 1), (1, 1), (0, 0)),
        ],
        "features_stage_settings": ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0)),
    },
)
registry.register_alias(
    "mobilenet_v4_l",
    MobileNet_v4,
    config={
        "dropout": 0.2,
        "stochastic_depth_prob": 0.35,
        "stem_settings": ConvNormActConfig(0, 24, (3, 3), (2, 2), (1, 1)),
        "net_settings": [
            # Stage 1
            InvertedResidualConfig(24, 48, (3, 3), (2, 2), (1, 1), 4.0, False),
            # Stage 2
            UniversalInvertedBottleneckConfig(48, 96, 4.0, (3, 3), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(96, 96, 4.0, (3, 3), (3, 3), (1, 1), True),
            # Stage 3
            UniversalInvertedBottleneckConfig(96, 192, 4.0, (3, 3), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), None, (1, 1), True),
            # Stage 4
            UniversalInvertedBottleneckConfig(192, 512, 4.0, (5, 5), (5, 5), (2, 2), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (3, 3), (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
            UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
        ],
        "last_stage_settings": [
            ConvNormActConfig(512, 960, (1, 1), (1, 1), (0, 0)),
        ],
        "features_stage_settings": ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0)),
    },
)

registry.register_weights(
    "mobilenet_v4_s_il-common",
    {
        "description": "MobileNet v4 small model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 11.5,
                "sha256": "faa6cb3adae59739892d28184a6746fe21df79599c0b09b66b5db745645587c3",
            }
        },
        "net": {"network": "mobilenet_v4_s", "tag": "il-common"},
    },
)
registry.register_weights(
    "mobilenet_v4_m_il-common",
    {
        "description": "MobileNet v4 medium model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 34.4,
                "sha256": "b34c25a14251084cead2c259e627ff00d6bbd2e23d841fc112dacae469ce1d8d",
            }
        },
        "net": {"network": "mobilenet_v4_m", "tag": "il-common"},
    },
)
