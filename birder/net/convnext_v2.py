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
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import PreTrainEncoder
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
class ConvNeXt_v2(DetectorBackbone, PreTrainEncoder):
    default_size = 224
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        in_channels: list[int] = self.config["in_channels"]
        num_layers: list[int] = self.config["num_layers"]
        stochastic_depth_prob: float = self.config["stochastic_depth_prob"]
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
        for idx, (i, out, n) in enumerate(zip(in_channels, out_channels, num_layers)):
            layers = []

            # Bottlenecks
            for _ in range(n):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                layers.append(ConvNeXtBlock(i, sd_prob))
                stage_block_id += 1

            # Down sampling
            if out != -1:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(i, eps=1e-6),
                        nn.Conv2d(i, out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                    )
                )

            stages[f"stage{idx+1}"] = nn.Sequential(*layers)
            return_channels.append(out if out != -1 else i)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(in_channels[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = in_channels[-1]
        self.classifier = self.create_classifier()

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

    def masked_encoding(
        self, x: torch.Tensor, mask_ratio: float, _mask_token: Optional[torch.Tensor] = None
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
        scale = 2**3
        assert len(mask.shape) == 2

        p = int(mask.shape[1] ** 0.5)
        upscale_mask = mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
        upscale_mask = upscale_mask.unsqueeze(1).type_as(x)

        x = self.stem(x)
        x *= 1.0 - upscale_mask
        x = self.body(x)

        return (x, mask)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias(
    "convnext_v2_atto",
    ConvNeXt_v2,
    config={"in_channels": [40, 80, 160, 320], "num_layers": [2, 2, 6, 2], "stochastic_depth_prob": 0.0},
)
registry.register_alias(
    "convnext_v2_femto",
    ConvNeXt_v2,
    config={"in_channels": [48, 96, 192, 384], "num_layers": [2, 2, 6, 2], "stochastic_depth_prob": 0.0},
)
registry.register_alias(
    "convnext_v2_pico",
    ConvNeXt_v2,
    config={"in_channels": [64, 128, 256, 512], "num_layers": [2, 2, 6, 2], "stochastic_depth_prob": 0.0},
)
registry.register_alias(
    "convnext_v2_nano",
    ConvNeXt_v2,
    config={"in_channels": [80, 160, 320, 640], "num_layers": [2, 2, 8, 2], "stochastic_depth_prob": 0.1},
)
registry.register_alias(
    "convnext_v2_tiny",
    ConvNeXt_v2,
    config={"in_channels": [96, 192, 384, 768], "num_layers": [3, 3, 9, 3], "stochastic_depth_prob": 0.2},
)
registry.register_alias(
    "convnext_v2_base",
    ConvNeXt_v2,
    config={"in_channels": [128, 256, 512, 1024], "num_layers": [3, 3, 27, 3], "stochastic_depth_prob": 0.1},
)
registry.register_alias(
    "convnext_v2_large",
    ConvNeXt_v2,
    config={"in_channels": [192, 384, 768, 1536], "num_layers": [3, 3, 27, 3], "stochastic_depth_prob": 0.2},
)
registry.register_alias(
    "convnext_v2_huge",
    ConvNeXt_v2,
    config={"in_channels": [352, 704, 1408, 2816], "num_layers": [3, 3, 27, 3], "stochastic_depth_prob": 0.3},
)

registry.register_weights(
    "convnext_v2_atto_il-common",
    {
        "description": "ConvNeXt v2 nano model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.4,
                "sha256": "2bf74e6f11a04f4e60970c7a8b2bbb239c5ddb7070f6c8b4978c310137023244",
            }
        },
        "net": {"network": "convnext_v2_atto", "tag": "il-common"},
    },
)
