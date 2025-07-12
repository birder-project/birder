"""
HGNet v2, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hgnet.py

Reference document:
https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.hgnet_v1 import Conv2dBNActivation
from birder.net.hgnet_v1 import HighPerfGPUStage
from birder.net.hgnet_v1 import LearnableAffineBlock


class Stem(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, use_lab: bool) -> None:
        super().__init__()
        self.stem1 = Conv2dBNActivation(
            in_channels,
            mid_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=1,
            use_act=True,
            use_lab=use_lab,
        )
        self.stem2a = Conv2dBNActivation(
            mid_channels,
            mid_channels // 2,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            use_act=True,
            use_lab=use_lab,
        )
        self.stem2b = Conv2dBNActivation(
            mid_channels // 2,
            mid_channels,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            use_act=True,
            use_lab=use_lab,
        )
        self.stem3 = Conv2dBNActivation(
            mid_channels * 2,
            mid_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=1,
            use_act=True,
            use_lab=use_lab,
        )
        self.stem4 = Conv2dBNActivation(
            mid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            use_act=True,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x1 = self.pool(x)
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)

        x = torch.concat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


# pylint: disable=invalid-name
class HGNet_v2(DetectorBackbone):
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

        use_ese = False
        feature_dim = 2048
        stem_channels: list[int] = self.config["stem_channels"]
        stage1: list[int | bool] = self.config["stage1"]
        stage2: list[int | bool] = self.config["stage2"]
        stage3: list[int | bool] = self.config["stage3"]
        stage4: list[int | bool] = self.config["stage4"]
        use_lab: bool = self.config["use_lab"]
        drop_path_rate: float = self.config["drop_path_rate"]

        stages_cfg = [stage1, stage2, stage3, stage4]
        block_depths = [c[3] for c in stages_cfg]

        self.stem = Stem(self.input_channels, stem_channels[0], stem_channels[1], use_lab=use_lab)

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
    "hgnet_v2_b0",
    HGNet_v2,
    config={
        "stem_channels": [16, 16],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
        "use_lab": True,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v2_b1",
    HGNet_v2,
    config={
        "stem_channels": [24, 32],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
        "use_lab": True,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v2_b2",
    HGNet_v2,
    config={
        "stem_channels": [24, 32],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 768, 3, True, True, 5, 4],
        "stage4": [768, 256, 1536, 1, True, True, 5, 4],
        "use_lab": True,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v2_b3",
    HGNet_v2,
    config={
        "stem_channels": [24, 32],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [32, 32, 128, 1, False, False, 3, 5],
        "stage2": [128, 64, 512, 1, True, False, 3, 5],
        "stage3": [512, 128, 1024, 3, True, True, 5, 5],
        "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
        "use_lab": True,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v2_b4",
    HGNet_v2,
    config={
        "stem_channels": [32, 48],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [48, 48, 128, 1, False, False, 3, 6],
        "stage2": [128, 96, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        "use_lab": False,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v2_b5",
    HGNet_v2,
    config={
        "stem_channels": [32, 64],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [64, 64, 128, 1, False, False, 3, 6],
        "stage2": [128, 128, 512, 2, True, False, 3, 6],
        "stage3": [512, 256, 1024, 5, True, True, 5, 6],
        "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
        "use_lab": False,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "hgnet_v2_b6",
    HGNet_v2,
    config={
        "stem_channels": [48, 96],
        # in, mid, out, blocks, downsample, light_block, kernel_size, num_layers
        "stage1": [96, 96, 192, 2, False, False, 3, 6],
        "stage2": [192, 192, 512, 3, True, False, 3, 6],
        "stage3": [512, 384, 1024, 6, True, True, 5, 6],
        "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
        "use_lab": False,
        "drop_path_rate": 0.0,
    },
)

registry.register_weights(
    "hgnet_v2_b0_il-common",
    {
        "description": "HGNet v2 B0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 18.1,
                "sha256": "bb1a45bb4f7612ca60d524c5c31e42b445c75bfac4142b029f0523c7e6a697ff",
            }
        },
        "net": {"network": "hgnet_v2_b0", "tag": "il-common"},
    },
)
