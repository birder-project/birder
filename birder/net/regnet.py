"""
RegNet, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py

Paper "Designing Network Design Spaces", https://arxiv.org/abs/2003.13678
"""

# Reference license: BSD 3-Clause

import math
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Iterator
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.base import make_divisible


class BlockParams:
    def __init__(
        self,
        depths: list[int],
        widths: list[int],
        group_widths: list[int],
        bottleneck_multipliers: list[float],
        strides: list[int],
        se_ratio: float,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> "BlockParams":
        """
        Programmatically compute all the per-block settings, given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - w_a is the width progression slope
        - w_0 is the initial width
        - w_m is the width stepping in the log space

        In other terms
        log(block_width) = log(w_0) + w_m * block_capacity,
        with bock_capacity ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.
        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8  # pylint: disable=invalid-name
        STRIDE = 2  # pylint: disable=invalid-name

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")

        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(block_widths + [0], [0] + block_widths, block_widths + [0], [0] + block_widths)
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        (stage_widths, group_widths) = cls._adjust_widths_groups_compatibility(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self) -> Iterator[tuple[int, int, int, int, float]]:
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibility(
        stage_widths: list[int], bottleneck_ratios: list[float], group_widths: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """

        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class BottleneckTransform(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: tuple[int, int],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        width_se_out = int(round(se_ratio * width_in))

        if width_se_out == 0:
            se_block = nn.Identity()
        else:
            se_block = SqueezeExcitation(input_channels=w_b, squeeze_channels=width_se_out, activation=activation_layer)

        self.block = nn.Sequential(
            Conv2dNormActivation(
                width_in,
                w_b,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=activation_layer,
                bias=False,
            ),
            Conv2dNormActivation(
                w_b,
                w_b,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=g,
                activation_layer=activation_layer,
                bias=False,
            ),
            se_block,
            Conv2dNormActivation(
                w_b, width_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False, activation_layer=None
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBottleneckBlock(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: tuple[int, int],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> None:
        super().__init__()

        if width_in != width_out or stride[0] != 1 or stride[1] != 1:
            self.proj = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=(1, 1),
                stride=stride,
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            )

        else:
            self.proj = nn.Identity()

        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride=stride,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            se_ratio=se_ratio,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.f(x)
        x = self.relu(x)

        return x


class AnyStage(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: tuple[int, int],
        depth: int,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(depth):
            if i == 0:
                in_ch = width_in
                cur_stride = stride
            else:
                in_ch = width_out
                cur_stride = (1, 1)

            layers.append(
                ResBottleneckBlock(
                    in_ch,
                    width_out,
                    stride=cur_stride,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                    se_ratio=se_ratio,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RegNet(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.block\.(\d+)"

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

        stem_width = 32
        bottleneck_multiplier = 1.0
        depth: int = self.config["depth"]
        w_0: int = self.config["w_0"]
        w_a: float = self.config["w_a"]
        w_m: float = self.config["w_m"]
        group_width: int = self.config["group_width"]
        se_ratio: float = self.config["se_ratio"]

        block_params = BlockParams.from_init_params(depth, w_0, w_a, w_m, group_width, bottleneck_multiplier, se_ratio)

        self.stem = Conv2dNormActivation(
            self.input_channels, stem_width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        current_width = stem_width
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        i = 0
        for (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in block_params._get_expanded_params():
            stages[f"stage{i+1}"] = AnyStage(
                current_width, width_out, (stride, stride), depth, group_width, bottleneck_multiplier, se_ratio
            )
            return_channels.append(width_out)
            current_width = width_out
            i += 1

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = current_width
        self.classifier = self.create_classifier()

        self.stem_stride = 2
        self.stem_width = stem_width
        self.encoding_size = current_width
        decoder_block = partial(
            BottleneckTransform,
            stride=(1, 1),
            group_width=64,
            bottleneck_multiplier=1.0,
            se_ratio=se_ratio,
        )
        self.decoder_block = lambda x: decoder_block(x, x)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
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

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        x = self.stem(x)
        x = mask_tensor(x, mask, patch_factor=self.max_stride // self.stem_stride, mask_token=mask_token)
        x = self.body(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            result["features"] = x
        if return_keys in ("all", "embedding"):
            result["embedding"] = self.features(x)

        return result

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


# RegNet X
registry.register_alias(
    "regnet_x_200m",
    RegNet,
    config={"depth": 13, "w_0": 24, "w_a": 36.44, "w_m": 2.49, "group_width": 8, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_400m",
    RegNet,
    config={"depth": 22, "w_0": 24, "w_a": 24.48, "w_m": 2.54, "group_width": 16, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_600m",
    RegNet,
    config={"depth": 16, "w_0": 48, "w_a": 36.97, "w_m": 2.24, "group_width": 24, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_800m",
    RegNet,
    config={"depth": 16, "w_0": 56, "w_a": 35.73, "w_m": 2.28, "group_width": 16, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_1_6g",
    RegNet,
    config={"depth": 18, "w_0": 80, "w_a": 34.01, "w_m": 2.25, "group_width": 24, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_3_2g",
    RegNet,
    config={"depth": 25, "w_0": 88, "w_a": 26.31, "w_m": 2.25, "group_width": 48, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_4g",
    RegNet,
    config={"depth": 23, "w_0": 96, "w_a": 38.65, "w_m": 2.43, "group_width": 40, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_6_4g",
    RegNet,
    config={"depth": 17, "w_0": 184, "w_a": 60.83, "w_m": 2.07, "group_width": 56, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_8g",
    RegNet,
    config={"depth": 23, "w_0": 80, "w_a": 49.56, "w_m": 2.88, "group_width": 120, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_12g",
    RegNet,
    config={"depth": 19, "w_0": 168, "w_a": 73.36, "w_m": 2.37, "group_width": 112, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_16g",
    RegNet,
    config={"depth": 22, "w_0": 216, "w_a": 55.59, "w_m": 2.1, "group_width": 128, "se_ratio": 0.0},
)
registry.register_alias(
    "regnet_x_32g",
    RegNet,
    config={"depth": 23, "w_0": 320, "w_a": 69.86, "w_m": 2.0, "group_width": 168, "se_ratio": 0.0},
)

# RegNet Y
registry.register_alias(
    "regnet_y_200m",
    RegNet,
    config={"depth": 13, "w_0": 24, "w_a": 36.44, "w_m": 2.49, "group_width": 8, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_400m",
    RegNet,
    config={"depth": 16, "w_0": 48, "w_a": 27.89, "w_m": 2.09, "group_width": 8, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_600m",
    RegNet,
    config={"depth": 15, "w_0": 48, "w_a": 32.54, "w_m": 2.32, "group_width": 16, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_800m",
    RegNet,
    config={"depth": 14, "w_0": 56, "w_a": 38.84, "w_m": 2.4, "group_width": 16, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_1_6g",
    RegNet,
    config={"depth": 27, "w_0": 48, "w_a": 20.71, "w_m": 2.65, "group_width": 24, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_3_2g",
    RegNet,
    config={"depth": 21, "w_0": 80, "w_a": 42.63, "w_m": 2.66, "group_width": 24, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_4g",
    RegNet,
    config={"depth": 22, "w_0": 96, "w_a": 31.41, "w_m": 2.24, "group_width": 64, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_6_4g",
    RegNet,
    config={"depth": 25, "w_0": 112, "w_a": 33.22, "w_m": 2.27, "group_width": 72, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_8g",
    RegNet,
    config={"depth": 17, "w_0": 192, "w_a": 76.82, "w_m": 2.19, "group_width": 56, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_12g",
    RegNet,
    config={"depth": 19, "w_0": 168, "w_a": 73.36, "w_m": 2.37, "group_width": 112, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_16g",
    RegNet,
    config={"depth": 18, "w_0": 200, "w_a": 106.23, "w_m": 2.48, "group_width": 112, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_32g",
    RegNet,
    config={"depth": 20, "w_0": 232, "w_a": 115.89, "w_m": 2.53, "group_width": 232, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_64g",
    RegNet,
    config={"depth": 20, "w_0": 352, "w_a": 147.48, "w_m": 2.4, "group_width": 328, "se_ratio": 0.25},
)
registry.register_alias(
    "regnet_y_128g",
    RegNet,
    config={"depth": 27, "w_0": 456, "w_a": 160.83, "w_m": 2.52, "group_width": 264, "se_ratio": 0.25},
)

registry.register_weights(
    "regnet_x_400m_il-common",
    {
        "description": "RegNet X 400m model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 20.3,
                "sha256": "c20ea88412fd2d85e28426dcb8300d38bc0fea856e7d6afff2a616fa0363c06d",
            }
        },
        "net": {"network": "regnet_x_400m", "tag": "il-common"},
    },
)

registry.register_weights(
    "regnet_y_200m_il-common",
    {
        "description": "RegNet Y 200m model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 11.4,
                "sha256": "0761351fc6624e476b59828efb88859f3667943aca918f732817b2d35a108884",
            }
        },
        "net": {"network": "regnet_y_200m", "tag": "il-common"},
    },
)
registry.register_weights(
    "regnet_y_400m_il-common",
    {
        "description": "RegNet Y 400m model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 15.8,
                "sha256": "20cd73f54aff3fddff410cba7686da8e4768f74c7a4981cfde4217e7e58eb71b",
            }
        },
        "net": {"network": "regnet_y_400m", "tag": "il-common"},
    },
)
registry.register_weights(
    "regnet_y_600m_il-common",
    {
        "description": "RegNet Y 600m model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.9,
                "sha256": "7ff9dc09d0f3216d85fe739fd0977b98a96e394d9a8916afa5454aa8fedda4c6",
            }
        },
        "net": {"network": "regnet_y_600m", "tag": "il-common"},
    },
)
registry.register_weights(
    "regnet_y_1_6g_il-common",
    {
        "description": "RegNet Y 1.6g model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 41.0,
                "sha256": "c2f8b17559a7d55efd1119352612a93063b393e0f05e8eb7ab0b5f637344cacc",
            }
        },
        "net": {"network": "regnet_y_1_6g", "tag": "il-common"},
    },
)
registry.register_weights(
    "regnet_y_1_6g_eu-common",
    {
        "url": "https://huggingface.co/birder-project/regnet_y_1_6g_eu-common/resolve/main",
        "description": "RegNet Y 1.6g model trained on the eu-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 42.2,
                "sha256": "a984384cbdd41d47ec179b24b49b3a9f097cc71d120caea27c2cce15b0499287",
            }
        },
        "net": {"network": "regnet_y_1_6g", "tag": "eu-common"},
    },
)
registry.register_weights(
    "regnet_y_8g_intermediate-eu-common",
    {
        "url": "https://huggingface.co/birder-project/regnet_y_8g_intermediate-eu-common/resolve/main",
        "description": "RegNet Y 8g model with intermediate training, then fine-tuned on the eu-common dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 148.5,
                "sha256": "96ac500880672ab4cf0dd6de92742a131c238ca49e6fb6040de543528c071147",
            }
        },
        "net": {"network": "regnet_y_8g", "tag": "intermediate-eu-common"},
    },
)
