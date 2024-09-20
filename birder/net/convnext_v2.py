"""
ConvNeXt v2, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py

Paper "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders",
https://arxiv.org/abs/2301.00808
"""

# Reference license: BSD 3-Clause and Apache-2.0

from functools import partial
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
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


# pylint: disable=invalid-name,too-many-branches
class ConvNeXt_v2(PreTrainEncoder):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        net_param = int(self.net_param)

        if net_param == 0:
            # Atto
            in_channels = [40, 80, 160, 320]
            out_channels = [80, 160, 320, -1]
            num_layers = [2, 2, 6, 2]
            stochastic_depth_prob = 0.0

        elif net_param == 1:
            # Femto
            in_channels = [48, 96, 192, 384]
            out_channels = [96, 192, 384, -1]
            num_layers = [2, 2, 6, 2]
            stochastic_depth_prob = 0.0

        elif net_param == 2:
            # Pico
            in_channels = [64, 128, 256, 512]
            out_channels = [128, 256, 512, -1]
            num_layers = [2, 2, 6, 2]
            stochastic_depth_prob = 0.0

        elif net_param == 3:
            # Nano
            in_channels = [80, 160, 320, 640]
            out_channels = [160, 320, 640, -1]
            num_layers = [2, 2, 8, 2]
            stochastic_depth_prob = 0.1

        elif net_param == 4:
            # Tiny
            in_channels = [96, 192, 384, 768]
            out_channels = [192, 384, 768, -1]
            num_layers = [3, 3, 9, 3]
            stochastic_depth_prob = 0.2

        elif net_param == 5:
            # Base
            in_channels = [128, 256, 512, 1024]
            out_channels = [256, 512, 1024, -1]
            num_layers = [3, 3, 27, 3]
            stochastic_depth_prob = 0.1

        elif net_param == 6:
            # Large
            in_channels = [192, 384, 768, 1536]
            out_channels = [384, 768, 1536, -1]
            num_layers = [3, 3, 27, 3]
            stochastic_depth_prob = 0.2

        elif net_param == 7:
            # Huge
            in_channels = [352, 704, 1408, 2816]
            out_channels = [704, 1408, 2816, -1]
            num_layers = [3, 3, 27, 3]
            stochastic_depth_prob = 0.3

        else:
            raise ValueError(f"net_param = {net_param} not supported")

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

        layers = []
        total_stage_blocks = sum(num_layers)
        stage_block_id = 0
        for i, out, n in zip(in_channels, out_channels, num_layers):
            # Bottlenecks
            stage = []
            for _ in range(n):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(ConvNeXtBlock(i, sd_prob))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

            # Down sampling
            if out != -1:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(i, eps=1e-6),
                        nn.Conv2d(i, out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                    )
                )

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(in_channels[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.embedding_size = in_channels[-1]
        self.classifier = self.create_classifier()

        self.encoding_size = in_channels[-1]
        self.decoder_block = partial(ConvNeXtBlock, stochastic_depth_prob=0)

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if isinstance(self.classifier, nn.Linear) is True:
            self.classifier.weight.data.mul_(0.001)
            self.classifier.bias.data.mul_(0.001)

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


registry.register_alias("convnext_v2_atto", ConvNeXt_v2, 0)
registry.register_alias("convnext_v2_femto", ConvNeXt_v2, 1)
registry.register_alias("convnext_v2_pico", ConvNeXt_v2, 2)
registry.register_alias("convnext_v2_nano", ConvNeXt_v2, 3)
registry.register_alias("convnext_v2_tiny", ConvNeXt_v2, 4)
registry.register_alias("convnext_v2_base", ConvNeXt_v2, 5)
registry.register_alias("convnext_v2_large", ConvNeXt_v2, 6)
registry.register_alias("convnext_v2_huge", ConvNeXt_v2, 7)
