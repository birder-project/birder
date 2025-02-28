"""
RegNet Z, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/regnet.py

Paper "Fast and Accurate Model Scaling", https://arxiv.org/abs/2103.06877
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from functools import partial
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import PreTrainEncoder
from birder.net.regnet import BlockParams
from birder.net.regnet import BottleneckTransform


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
            self.shortcut = False

        else:
            self.shortcut = True

        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride=stride,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            se_ratio=se_ratio,
            activation_layer=nn.SiLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is True:
            return x + self.f(x)

        return self.f(x)


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


# pylint: disable=invalid-name
class RegNet_Z(DetectorBackbone, PreTrainEncoder):
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
        bottleneck_multiplier = 4.0
        se_ratio = 0.25
        depth: int = self.config["depth"]
        w_0: int = self.config["w_0"]
        w_a: float = self.config["w_a"]
        w_m: float = self.config["w_m"]
        group_width: int = self.config["group_width"]
        num_features: int = self.config["num_features"]

        block_params = BlockParams.from_init_params(depth, w_0, w_a, w_m, group_width, bottleneck_multiplier, se_ratio)

        self.stem = Conv2dNormActivation(
            self.input_channels,
            stem_width,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=nn.SiLU,
            bias=False,
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
            Conv2dNormActivation(
                current_width, num_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=nn.SiLU
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

        self.encoding_size = num_features
        decoder_block = partial(
            BottleneckTransform,
            stride=(1, 1),
            group_width=64,
            bottleneck_multiplier=1.0,
            se_ratio=se_ratio,
            activation_layer=nn.SiLU,
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

    def masked_encoding(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        kept_mask_ratio: Optional[float] = None,
        mask_token: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        L = (x.shape[2] // 32) ** 2  # Patch size = 32
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # Un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Upsample mask
        scale = 2**4
        assert len(mask.shape) == 2

        p = int(mask.shape[1] ** 0.5)
        upscale_mask = mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
        upscale_mask = upscale_mask.unsqueeze(1).type_as(x)

        x = self.stem(x)
        x = x * (1.0 - upscale_mask)
        x = self.body(x)

        return (x, mask)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias(
    "regnet_z_500m",
    RegNet_Z,
    config={"depth": 21, "w_0": 16, "w_a": 10.7, "w_m": 2.51, "group_width": 4, "num_features": 1024},
)
registry.register_alias(
    "regnet_z_4g",
    RegNet_Z,
    config={"depth": 28, "w_0": 48, "w_a": 14.5, "w_m": 2.226, "group_width": 8, "num_features": 1536},
)

registry.register_weights(
    "regnet_z_500m_il-common",
    {
        "description": "RegNet Z 500m model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 25.1,
                "sha256": "b2cc5c9f5c5e4693d8fe12e2d5eddaa28ce25d9ea38e14ea67ec09706aa24ea9",
            }
        },
        "net": {"network": "regnet_z_500m", "tag": "il-common"},
    },
)
