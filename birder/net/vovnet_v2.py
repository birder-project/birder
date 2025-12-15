"""
VoVNet v2 (ESE-VoVNet), adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vovnet.py

Paper "CenterMask : Real-Time Anchor-Free Instance Segmentation", https://arxiv.org/abs/1911.06667
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class EffectiveSE(nn.Module):
    """
    Effective Squeeze-Excitation module from CenterMask paper
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean(dim=(2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * x_se.sigmoid()


class OSABlock(nn.Module):
    """
    One-Shot Aggregation block
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        layers_per_block: int,
        residual: bool,
        use_ese: bool,
    ) -> None:
        super().__init__()
        self.residual = residual

        mid_convs = []
        next_in = in_channels
        for _ in range(layers_per_block):
            mid_convs.append(
                Conv2dNormActivation(
                    next_in,
                    mid_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                )
            )
            next_in = mid_channels

        self.mid_convs = nn.ModuleList(mid_convs)

        agg_channels = in_channels + layers_per_block * mid_channels
        self.concat_conv = Conv2dNormActivation(
            agg_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        if use_ese is True:
            self.ese: nn.Module = EffectiveSE(out_channels)
        else:
            self.ese = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        features = [x]
        for conv in self.mid_convs:
            x = conv(x)
            features.append(x)

        x = torch.concat(features, dim=1)
        x = self.concat_conv(x)
        x = self.ese(x)

        if self.residual is True:
            x = x + identity

        return x


class OSAStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        blocks_per_stage: int,
        layers_per_block: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if downsample is True:
            layers.append(nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True))

        for i in range(blocks_per_stage):
            last_block = i == blocks_per_stage - 1
            block_in = in_channels if i == 0 else out_channels
            layers.append(
                OSABlock(
                    block_in,
                    mid_channels,
                    out_channels,
                    layers_per_block,
                    residual=i > 0,
                    use_ese=last_block,
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# pylint: disable=invalid-name
class VoVNet_v2(DetectorBackbone):
    block_group_regex = r"body\.stage(\d+)\.blocks\.(\d+)"

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

        stem_channels: list[int] = self.config["stem_channels"]
        stage_conv_channels: list[int] = self.config["stage_conv_channels"]
        stage_out_channels: list[int] = self.config["stage_out_channels"]
        blocks_per_stage: list[int] = self.config["blocks_per_stage"]
        layers_per_block: int = self.config["layers_per_block"]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                stem_channels[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            Conv2dNormActivation(
                stem_channels[0],
                stem_channels[1],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            Conv2dNormActivation(
                stem_channels[1],
                stem_channels[2],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        in_ch_list = [stem_channels[-1]] + stage_out_channels[:-1]

        for i in range(4):
            stages[f"stage{i + 1}"] = OSAStage(
                in_channels=in_ch_list[i],
                mid_channels=stage_conv_channels[i],
                out_channels=stage_out_channels[i],
                blocks_per_stage=blocks_per_stage[i],
                layers_per_block=layers_per_block,
                downsample=i > 0,
            )
            return_channels.append(stage_out_channels[i])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = stage_out_channels[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
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
    "vovnet_v2_19",
    VoVNet_v2,
    config={
        "stem_channels": [64, 64, 128],
        "stage_conv_channels": [128, 160, 192, 224],
        "stage_out_channels": [256, 512, 768, 1024],
        "blocks_per_stage": [1, 1, 1, 1],
        "layers_per_block": 3,
    },
)
registry.register_model_config(
    "vovnet_v2_39",
    VoVNet_v2,
    config={
        "stem_channels": [64, 64, 128],
        "stage_conv_channels": [128, 160, 192, 224],
        "stage_out_channels": [256, 512, 768, 1024],
        "blocks_per_stage": [1, 1, 2, 2],
        "layers_per_block": 5,
    },
)
registry.register_model_config(
    "vovnet_v2_57",
    VoVNet_v2,
    config={
        "stem_channels": [64, 64, 128],
        "stage_conv_channels": [128, 160, 192, 224],
        "stage_out_channels": [256, 512, 768, 1024],
        "blocks_per_stage": [1, 1, 4, 3],
        "layers_per_block": 5,
    },
)
registry.register_model_config(
    "vovnet_v2_99",
    VoVNet_v2,
    config={
        "stem_channels": [64, 64, 128],
        "stage_conv_channels": [128, 160, 192, 224],
        "stage_out_channels": [256, 512, 768, 1024],
        "blocks_per_stage": [1, 3, 9, 3],
        "layers_per_block": 5,
    },
)
