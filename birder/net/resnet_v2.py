"""
ResNet v2, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py

Paper "Identity Mappings in Deep Residual Networks", https://arxiv.org/abs/1603.05027
and
Paper "Squeeze-and-Excitation Networks", https://arxiv.org/abs/1709.01507
"""

# Reference license: Apache-2.0

from collections import OrderedDict
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


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int], bottle_neck: bool, squeeze_excitation: bool
    ) -> None:
        super().__init__()
        if bottle_neck is True:
            self.block1 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                Conv2dNormActivation(
                    in_channels,
                    out_channels // 4,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
                Conv2dNormActivation(
                    out_channels // 4,
                    out_channels // 4,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.Conv2d(
                    out_channels // 4,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
            )

        else:
            self.block1 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                Conv2dNormActivation(
                    in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            )

        self.use_projection = in_channels != out_channels or stride != (1, 1)
        if self.use_projection is True:
            self.block2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0), bias=False
            )
        else:
            self.block2 = nn.Identity()

        if squeeze_excitation is True:
            self.se = SqueezeExcitation(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for idx, module in enumerate(self.block1):
            x = module(x)
            if idx == 1 and self.use_projection is True:
                identity = x

        x = self.se(x)
        identity = self.block2(identity)

        return x + identity


class ResNet_v2(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage(\d+)\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        bottle_neck: bool = self.config["bottle_neck"]
        filter_list: list[int] = self.config["filter_list"]
        units: list[int] = self.config["units"]
        squeeze_excitation: bool = self.config.get("squeeze_excitation", False)

        assert len(units) + 1 == len(filter_list)
        num_unit = len(units)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                filter_list[0],
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # Generate body layers
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_unit):
            layers = []
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            layers.append(
                ResidualBlock(
                    filter_list[i],
                    filter_list[i + 1],
                    stride=stride,
                    bottle_neck=bottle_neck,
                    squeeze_excitation=squeeze_excitation,
                )
            )
            for _ in range(1, units[i]):
                layers.append(
                    ResidualBlock(
                        filter_list[i + 1],
                        filter_list[i + 1],
                        stride=(1, 1),
                        bottle_neck=bottle_neck,
                        squeeze_excitation=squeeze_excitation,
                    )
                )

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(filter_list[i + 1])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(filter_list[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = filter_list[-1]
        self.classifier = self.create_classifier()

        self.max_stride = 32
        self.stem_stride = 4
        self.stem_width = filter_list[0]
        self.feature_dim = filter_list[-1]

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
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

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

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)


registry.register_model_config(
    "resnet_v2_18",
    ResNet_v2,
    config={"bottle_neck": False, "filter_list": [64, 64, 128, 256, 512], "units": [2, 2, 2, 2]},
)
registry.register_model_config(
    "resnet_v2_34",
    ResNet_v2,
    config={"bottle_neck": False, "filter_list": [64, 64, 128, 256, 512], "units": [3, 4, 6, 3]},
)
registry.register_model_config(
    "resnet_v2_50",
    ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 4, 6, 3]},
)
registry.register_model_config(
    "resnet_v2_101",
    ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 4, 23, 3]},
)
registry.register_model_config(
    "resnet_v2_152",
    ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 8, 36, 3]},
)
registry.register_model_config(
    "resnet_v2_200",
    ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 24, 36, 3]},
)
registry.register_model_config(
    "resnet_v2_269",
    ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 30, 48, 8]},
)

# Squeeze-and-Excitation Networks
registry.register_model_config(
    "se_resnet_v2_18",
    ResNet_v2,
    config={
        "bottle_neck": False,
        "filter_list": [64, 64, 128, 256, 512],
        "units": [2, 2, 2, 2],
        "squeeze_excitation": True,
    },
)
registry.register_model_config(
    "se_resnet_v2_34",
    ResNet_v2,
    config={
        "bottle_neck": False,
        "filter_list": [64, 64, 128, 256, 512],
        "units": [3, 4, 6, 3],
        "squeeze_excitation": True,
    },
)
registry.register_model_config(
    "se_resnet_v2_50",
    ResNet_v2,
    config={
        "bottle_neck": True,
        "filter_list": [64, 256, 512, 1024, 2048],
        "units": [3, 4, 6, 3],
        "squeeze_excitation": True,
    },
)
registry.register_model_config(
    "se_resnet_v2_101",
    ResNet_v2,
    config={
        "bottle_neck": True,
        "filter_list": [64, 256, 512, 1024, 2048],
        "units": [3, 4, 23, 3],
        "squeeze_excitation": True,
    },
)
registry.register_model_config(
    "se_resnet_v2_152",
    ResNet_v2,
    config={
        "bottle_neck": True,
        "filter_list": [64, 256, 512, 1024, 2048],
        "units": [3, 8, 36, 3],
        "squeeze_excitation": True,
    },
)
registry.register_model_config(
    "se_resnet_v2_200",
    ResNet_v2,
    config={
        "bottle_neck": True,
        "filter_list": [64, 256, 512, 1024, 2048],
        "units": [3, 24, 36, 3],
        "squeeze_excitation": True,
    },
)
registry.register_model_config(
    "se_resnet_v2_269",
    ResNet_v2,
    config={
        "bottle_neck": True,
        "filter_list": [64, 256, 512, 1024, 2048],
        "units": [3, 30, 48, 8],
        "squeeze_excitation": True,
    },
)

registry.register_weights(
    "resnet_v2_50_inat21-256px",
    {
        "url": "https://huggingface.co/birder-project/resnet_v2_50_inat21/resolve/main",
        "description": "ResNet v2 50 model trained on the iNaturalist 2021 dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 169.0,
                "sha256": "2a34cdaa704f56cb231fb3726e35f90439fc6ab757f6578263e07f2c91b72e6a",
            }
        },
        "net": {"network": "resnet_v2_50", "tag": "inat21-256px"},
    },
)
registry.register_weights(
    "resnet_v2_50_inat21",
    {
        "url": "https://huggingface.co/birder-project/resnet_v2_50_inat21/resolve/main",
        "description": "ResNet v2 50 model trained on the iNaturalist 2021 dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 169.0,
                "sha256": "72a7cfd93a2225461a9c43701c63e93e71f12c0f857e3d087e64ecede6bb1067",
            }
        },
        "net": {"network": "resnet_v2_50", "tag": "inat21"},
    },
)
