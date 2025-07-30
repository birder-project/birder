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
        assert hasattr(self.backbone, "features") is True

        fixed_gem_pool_3: type[nn.Module] = partial(FixedGeMPool2d, 3)  # type: ignore[assignment]
        self.backbone = training_utils.replace_module(self.backbone, nn.AdaptiveAvgPool2d, fixed_gem_pool_3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.normalize(x)

        return x
