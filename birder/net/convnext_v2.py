"""
ConvNeXt v2, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py

Paper "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders",
https://arxiv.org/abs/2301.00808
"""

# Reference license: BSD 3-Clause and Apache-2.0

from collections import OrderedDict
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.convnext_v1 import LayerNorm2d


class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)  # pylint: disable=not-callable
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)

        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding=(3, 3),
                groups=channels,
                bias=True,
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, 4 * channels),  # Same as 1x1 conv
            nn.GELU(),
            GRN(4 * channels),
            nn.Linear(4 * channels, channels),  # Same as 1x1 conv
            Permute([0, 3, 1, 2]),
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x)
        x = self.stochastic_depth(x)
        x += identity

        return x


# pylint: disable=invalid-name
class ConvNeXt_v2(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
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

        in_channels: list[int] = self.config["in_channels"]
        num_layers: list[int] = self.config["num_layers"]
        drop_path_rate: float = self.config["drop_path_rate"]
        out_channels = in_channels[1:] + [-1]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            in_channels[0],
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0),
            bias=True,
            norm_layer=LayerNorm2d,
            activation_layer=None,
        )

        total_stage_blocks = sum(num_layers)
        stage_block_id = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        layers = []
        for idx, (i, out, n) in enumerate(zip(in_channels, out_channels, num_layers)):
            # Bottlenecks
            for _ in range(n):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = drop_path_rate * stage_block_id / (total_stage_blocks - 1.0)
                layers.append(ConvNeXtBlock(i, sd_prob))
                stage_block_id += 1

            stages[f"stage{idx+1}"] = nn.Sequential(*layers)
            return_channels.append(i)
            layers = []

            # Down sampling
            if out != -1:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(i, eps=1e-6),
                        nn.Conv2d(i, out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                    )
                )

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(in_channels[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = in_channels[-1]
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = in_channels[0]
        self.encoding_size = in_channels[-1]
        self.decoder_block = partial(ConvNeXtBlock, stochastic_depth_prob=0)

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.mul_(0.001)
            self.classifier.bias.data.mul_(0.001)

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


registry.register_alias(
    "convnext_v2_atto",
    ConvNeXt_v2,
    config={"in_channels": [40, 80, 160, 320], "num_layers": [2, 2, 6, 2], "drop_path_rate": 0.0},
)
registry.register_alias(
    "convnext_v2_femto",
    ConvNeXt_v2,
    config={"in_channels": [48, 96, 192, 384], "num_layers": [2, 2, 6, 2], "drop_path_rate": 0.0},
)
registry.register_alias(
    "convnext_v2_pico",
    ConvNeXt_v2,
    config={"in_channels": [64, 128, 256, 512], "num_layers": [2, 2, 6, 2], "drop_path_rate": 0.0},
)
registry.register_alias(
    "convnext_v2_nano",
    ConvNeXt_v2,
    config={"in_channels": [80, 160, 320, 640], "num_layers": [2, 2, 8, 2], "drop_path_rate": 0.1},
)
registry.register_alias(
    "convnext_v2_tiny",
    ConvNeXt_v2,
    config={"in_channels": [96, 192, 384, 768], "num_layers": [3, 3, 9, 3], "drop_path_rate": 0.2},
)
registry.register_alias(
    "convnext_v2_small",  # Not in the original v2, taken from v1
    ConvNeXt_v2,
    config={"in_channels": [96, 192, 384, 768], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.1},
)
registry.register_alias(
    "convnext_v2_base",
    ConvNeXt_v2,
    config={"in_channels": [128, 256, 512, 1024], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.1},
)
registry.register_alias(
    "convnext_v2_large",
    ConvNeXt_v2,
    config={"in_channels": [192, 384, 768, 1536], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.2},
)
registry.register_alias(
    "convnext_v2_huge",
    ConvNeXt_v2,
    config={"in_channels": [352, 704, 1408, 2816], "num_layers": [3, 3, 27, 3], "drop_path_rate": 0.3},
)

registry.register_weights(
    "convnext_v2_atto_il-common",
    {
        "description": "ConvNeXt v2 nano model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.4,
                "sha256": "b001b4d71c74966b530f847490edf1c4c7d769a34fa29aa96238602e3248a2a2",
            }
        },
        "net": {"network": "convnext_v2_atto", "tag": "il-common"},
    },
)
registry.register_weights(
    "convnext_v2_tiny_intermediate-il-common",
    {
        "url": "https://huggingface.co/birder-project/convnext_v2_tiny_intermediate-il-common/resolve/main",
        "description": "ConvNeXt v2 tiny model with intermediate training, then fine-tuned on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 107.5,
                "sha256": "590b74dbf39aa1d6356b9001d5505c00ac97b7399fa994efc58c7e14fd4f668a",
            }
        },
        "net": {"network": "convnext_v2_tiny", "tag": "intermediate-il-common"},
    },
)
registry.register_weights(
    "convnext_v2_tiny_intermediate-eu-common",
    {
        "url": "https://huggingface.co/birder-project/convnext_v2_tiny_intermediate-eu-common/resolve/main",
        "description": "ConvNeXt v2 tiny model with intermediate training, then fine-tuned on the eu-common dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 108.5,
                "sha256": "a0da4037acda6c50c9032a6462a810082aa79193bdec26203d98fdfdad699fc4",
            }
        },
        "net": {"network": "convnext_v2_tiny", "tag": "intermediate-eu-common"},
    },
)
