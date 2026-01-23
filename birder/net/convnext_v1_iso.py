"""
ConvNeXt v1 Isotropic, adapted from
https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext_isotropic.py

Paper "A ConvNet for the 2020s", https://arxiv.org/abs/2201.03545
"""

# Reference license: MIT

from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import LayerNorm2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.base import normalize_out_indices


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, stochastic_depth_prob: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=channels),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
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
class ConvNeXt_v1_Isotropic(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.(\d+)"

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

        patch_size = 16
        dim: int = self.config["dim"]
        num_layers: int = self.config["num_layers"]
        out_indices: Optional[list[int]] = self.config.get("out_indices", None)
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(self.size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(self.size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        self.patch_size = patch_size
        self.out_indices = normalize_out_indices(out_indices, num_layers)

        self.stem = nn.Conv2d(
            self.input_channels,
            dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        )

        layers = []
        for idx in range(num_layers):
            # Adjust stochastic depth probability based on the depth of the stage block
            sd_prob = drop_path_rate * idx / (num_layers - 1.0)
            layers.append(ConvNeXtBlock(dim, sd_prob))

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(dim, eps=1e-6),
            nn.Flatten(1),
        )

        num_return_stages = len(self.out_indices) if self.out_indices is not None else 1
        self.return_stages = [f"stage{stage_idx + 1}" for stage_idx in range(num_return_stages)]
        self.return_channels = [dim] * num_return_stages
        self.embedding_size = dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = dim
        self.encoding_size = dim
        self.decoder_block = partial(ConvNeXtBlock, stochastic_depth_prob=0)

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        if self.out_indices is None:
            x = self.body(x)
            return {self.return_stages[0]: x}

        stage_num = 0
        out: dict[str, torch.Tensor] = {}
        for idx, module in enumerate(self.body.children()):
            x = module(x)
            if idx in self.out_indices:
                out[self.return_stages[stage_num]] = x
                stage_num += 1

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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        super().adjust_size(new_size)


registry.register_model_config(
    "convnext_v1_iso_small",
    ConvNeXt_v1_Isotropic,
    config={"dim": 384, "num_layers": 18, "drop_path_rate": 0.1},
)
registry.register_model_config(
    "convnext_v1_iso_base",
    ConvNeXt_v1_Isotropic,
    config={"in_channels": 768, "num_layers": 18, "drop_path_rate": 0.2},
)
registry.register_model_config(
    "convnext_v1_iso_large",
    ConvNeXt_v1_Isotropic,
    config={"in_channels": 1024, "num_layers": 36, "drop_path_rate": 0.5},
)
