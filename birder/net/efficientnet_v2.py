"""
EfficientNet v2, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

Paper "EfficientNetV2: Smaller Models and Faster Training",
https://arxiv.org/abs/2104.00298
"""

# Reference license: BSD 3-Clause

import math
from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.base import make_divisible
from birder.net.efficientnet_v1 import MBConv


def adjust_channels(channels: int, width: float, min_value: Optional[int] = None) -> int:
    return make_divisible(channels * width, 8, min_value)


def adjust_depth(num_layers: int, depth: float) -> int:
    return int(math.ceil(num_layers * depth))


class FusedMBConv(nn.Module):
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

        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        expanded_channels = adjust_channels(in_channels, expand_ratio)
        if expanded_channels != in_channels:
            # Fused Expand
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    activation_layer=activation_layer,
                )
            )

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

        else:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    activation_layer=activation_layer,
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
class EfficientNet_v2(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
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

        width_coefficient = 1.0
        depth_coefficient = 1.0
        last_channels: int = self.config.get("last_channels", 1280)
        expand_ratio: list[int] = self.config["expand_ratio"]
        kernel_size: list[tuple[int, int]] = self.config["kernel_size"]
        strides: list[tuple[int, int]] = self.config["strides"]
        in_channels: list[int] = self.config["in_channels"]
        out_channels: list[int] = self.config["out_channels"]
        repeats: list[int] = self.config["repeats"]
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

                if i < 3:
                    layers.append(
                        FusedMBConv(
                            in_ch,
                            out_channels[i],
                            kernel_size=kernel_size[i],
                            stride=stride,
                            expand_ratio=expand_ratio[i],
                            stochastic_depth_prob=sd_prob,
                        )
                    )

                else:
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
                last_channels,
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
        self.embedding_size = last_channels
        self.classifier = self.create_classifier()

        self.stem_stride = 2
        self.stem_width = in_channels[0]
        self.encoding_size = out_channels[-1]

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
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


registry.register_model_config(
    "efficientnet_v2_s",
    EfficientNet_v2,  # input_resolution = 300 / 384 (train / eval)
    config={
        "expand_ratio": [1, 4, 4, 4, 6, 6],
        "kernel_size": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (2, 2)],
        "in_channels": [24, 24, 48, 64, 128, 160],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "repeats": [2, 4, 4, 6, 9, 15],
        "dropout_rate": 0.2,
    },
)
registry.register_model_config(
    "efficientnet_v2_m",
    EfficientNet_v2,  # input_resolution = 384 / 480 (train / eval)
    config={
        "expand_ratio": [1, 4, 4, 4, 6, 6, 6],
        "kernel_size": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (2, 2), (1, 1)],
        "in_channels": [24, 24, 48, 80, 160, 176, 304],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "repeats": [3, 5, 5, 7, 14, 18, 5],
        "dropout_rate": 0.3,
    },
)
registry.register_model_config(
    "efficientnet_v2_l",
    EfficientNet_v2,  # input_resolution = 384 / 480 (train / eval)
    config={
        "expand_ratio": [1, 4, 4, 4, 6, 6, 6],
        "kernel_size": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (2, 2), (1, 1)],
        "in_channels": [32, 32, 64, 96, 192, 224, 384],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "repeats": [4, 7, 7, 10, 19, 25, 7],
        "dropout_rate": 0.4,
    },
)
registry.register_model_config(
    "efficientnet_v2_xl",
    EfficientNet_v2,  # input_resolution = 384 / 512 (train / eval)
    config={
        "expand_ratio": [1, 4, 4, 4, 6, 6, 6],
        "kernel_size": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        "strides": [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (2, 2), (1, 1)],
        "in_channels": [32, 32, 64, 96, 192, 256, 512],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "repeats": [4, 8, 8, 16, 24, 32, 8],
        "dropout_rate": 0.4,
    },
)

registry.register_weights(
    "efficientnet_v2_s_il-common",
    {
        "description": "EfficientNet v2 small model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 79.7,
                "sha256": "abd0310951ae2dcc56792473bcefab49ab28971746322cd561d6c369503e4d6f",
            }
        },
        "net": {"network": "efficientnet_v2_s", "tag": "il-common"},
    },
)
registry.register_weights(
    "efficientnet_v2_s_arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/efficientnet_v2_s_arabian-peninsula/resolve/main",
        "description": "EfficientNet v2 small model trained on the arabian-peninsula dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 81.5,
                "sha256": "ad708a255d2e0fbcacc907d0b42a3881180a4e0a9b238dec409985b502dac62d",
            }
        },
        "net": {"network": "efficientnet_v2_s", "tag": "arabian-peninsula"},
    },
)
