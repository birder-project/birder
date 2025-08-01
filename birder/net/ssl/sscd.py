"""
Self-Supervised Descriptor for Image Copy Detection (SSCD), adapted from
https://github.com/facebookresearch/sscd-copy-detection/blob/main/sscd/models/model.py

Paper "A Self-Supervised Descriptor for Image Copy Detection",
https://arxiv.org/abs/2202.10261
"""

# Reference license: MIT

from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils
from birder.layers import FixedGeMPool2d
from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet


class SSCD(SSLBaseNet):
    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is None, "config not supported"

        fixed_gem_pool_3: type[nn.Module] = partial(FixedGeMPool2d, 3.0)  # type: ignore[assignment]
        self.backbone = training_utils.replace_module(self.backbone, nn.AdaptiveAvgPool2d, fixed_gem_pool_3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.normalize(x)

        return x


# Registered as classification model instead of SSL for cleaner API - functionally equivalent
registry.register_weights(
    "sscd_resnext_101_c1",
    {
        "url": "https://huggingface.co/birder-project/sscd_resnext_101_c1/resolve/main",
        "description": (
            "SSCD ResNeXt 101 model trained DISC for image copy detection. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (320, 320),
        "formats": {
            "pts": {
                "file_size": 169.9,
                "sha256": "8ccedaf9efc243be81d8a3d432d4bd8688fb91e71b43934f68557f45bb0adb3f",
            }
        },
        "net": {"network": "resnext_101", "tag": "c1_sscd"},
    },
)
