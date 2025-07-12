"""
HGNet v1, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hgnet.py

Reference document:
https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/docs/en/models/PP-HGNet_en.md
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float, bias_value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class Conv2dBNActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int,
        use_act: bool,
        use_lab: bool,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if use_act is True:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        if use_lab is True:
            self.lab = LearnableAffineBlock(scale_value=1.0, bias_value=0.0)
        else:
            self.lab = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)

        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], use_lab: bool) -> None:
        super().__init__()
        self.conv1 = Conv2dBNActivation(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            use_act=False,
            use_lab=False,
        )
        self.conv2 = Conv2dBNActivation(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class EseModule(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class Stem(nn.Module):
    def __init__(self, channel_list: list[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(
                Conv2dBNActivation(
                    channel_list[i],
                    channel_list[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2) if i == 0 else (1, 1),
                    padding=(1, 1),
                    groups=1,
                    use_act=True,
                    use_lab=False,
                )
            )

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)

        return x


class HighPerfGPUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: tuple[int, int],
        residual: bool,
        light_block: bool,
        use_lab: bool,
        use_ese: bool,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if light_block is True:
                self.layers.append(
                    LightConvBNAct(
                        in_channels if i == 0 else mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    Conv2dBNActivation(
                        in_channels if i == 0 else mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        stride=(1, 1),
                        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                        groups=1,
                        use_act=True,
                        use_lab=use_lab,
                    )
                )

        # Feature aggregation
        total_chs = in_channels + num_layers * mid_channels
        if use_ese is True:
            self.aggregation = nn.Sequential(
                Conv2dBNActivation(
                    total_chs,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    groups=1,
                    use_act=True,
                    use_lab=use_lab,
                ),
                EseModule(out_channels),
            )
        else:
            self.aggregation = nn.Sequential(
                Conv2dBNActivation(
                    total_chs,
                    out_channels // 2,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    groups=1,
                    use_act=True,
                    use_lab=use_lab,
                ),
                Conv2dBNActivation(
                    out_channels // 2,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    groups=1,
                    use_act=True,
                    use_lab=use_lab,
                ),
            )

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.concat(output, dim=1)
        x = self.aggregation(x)
        if self.residual is True:
            x = self.drop_path(x) + identity

        return x


class HighPerfGPUStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        depth: int,
        num_layers: int,
        downsample: bool,
        stride: tuple[int, int],
        kernel_size: tuple[int, int],
        light_block: bool,
        use_lab: bool,
        use_ese: bool,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = Conv2dBNActivation(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                groups=in_channels,
                use_act=False,
                use_lab=False,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(depth):
            blocks_list.append(
                HighPerfGPUBlock(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    num_layers,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    use_ese=use_ese,
                    drop_path=drop_path[i],
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


# pylint: disable=invalid-name
class HGNet_v1(DetectorBackbone):
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

        use_lab = False
        use_ese = True
        feature_dim = 2048
        stem_channels: list[int] = self.config["stem_channels"]
        stage1: list[int | bool] = self.config["stage1"]
        stage2: list[int | bool] = self.config["stage2"]
        stage3: list[int | bool] = self.config["stage3"]
        stage4: list[int | bool] = self.config["stage4"]
        drop_path_rate: float = self.config["drop_path_rate"]

        stages_cfg = [stage1, stage2, stage3, stage4]
        block_depths = [c[3] for c in stages_cfg]

        self.stem = Stem([self.input_channels] + stem_channels)

        # Stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(block_depths)).split(block_depths)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, stage_config in enumerate(stages_cfg):
            (in_ch, mid_ch, out_ch, blocks, downsample, light_block, k, num_layers) = stage_config
            stages[f"stage{i+1}"] = HighPerfGPUStage(
                in_channels=in_ch,
                mid_channels=mid_ch,
                out_channels=out_ch,
                depth=blocks,
                num_layers=num_layers,
                downsample=downsample,  # type: ignore[arg-type]
                stride=(2, 2),
                kernel_size=(k, k),
                light_block=light_block,  # type: ignore[arg-type]
                use_lab=use_lab,
                use_ese=use_ese,
                drop_path=dpr[i],
            )
            return_channels.append(out_ch)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(out_ch, feature_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            LearnableAffineBlock(scale_value=1.0, bias_value=0.0) if use_lab is True else nn.Identity(),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = feature_dim
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
    "hgnet_v1_tiny",
    HGNet_v1,
    config={
        "stem_channels": [48, 48, 96],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [96, 96, 224, 1, False, False, 3, 5],
        "stage2": [224, 128, 448, 1, True, False, 3, 5],
        "stage3": [448, 160, 512, 2, True, False, 3, 5],
        "stage4": [512, 192, 768, 1, True, False, 3, 5],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v1_small",
    HGNet_v1,
    config={
        "stem_channels": [64, 64, 128],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [128, 128, 256, 1, False, False, 3, 6],
        "stage2": [256, 160, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 768, 2, True, False, 3, 6],
        "stage4": [768, 224, 1024, 1, True, False, 3, 6],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v1_base",
    HGNet_v1,
    config={
        "stem_channels": [96, 96, 160],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [160, 192, 320, 1, False, False, 3, 7],
        "stage2": [320, 224, 640, 2, True, False, 3, 7],
        "stage3": [640, 256, 960, 3, True, False, 3, 7],
        "stage4": [960, 288, 1280, 2, True, False, 3, 7],
        "drop_path_rate": 0.0,
    },
)

registry.register_weights(
    "hgnet_v1_tiny_il-common",
    {
        "description": "HGNet v1 tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 51.4,
                "sha256": "c8895d04c8fb28b617e5c3c3c9186979ce480912873814c2cc2bb65cdf57e0f1",
            }
        },
        "net": {"network": "hgnet_v1_tiny", "tag": "il-common"},
    },
)
