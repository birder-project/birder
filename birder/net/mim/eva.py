"""
EVA-style masked image modeling.

Paper "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale",
https://arxiv.org/abs/2211.07636
"""

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.mim.base import MIMBaseNet


class EVA(MIMBaseNet):
    default_size = (224, 224)
    default_mask_ratio = 0.42

    def __init__(
        self,
        encoder: PreTrainEncoder,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
        mask_ratio: Optional[float] = None,
        min_mask_size: int = 1,
    ) -> None:
        super().__init__(encoder, config=config, size=size, mask_ratio=mask_ratio, min_mask_size=min_mask_size)
        assert self.config is not None, "must set config"
        assert isinstance(self.encoder, MaskedTokenRetentionMixin)

        teacher_dim: int = self.config["teacher_dim"]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.encoder.stem_width))
        self.predictor = nn.Linear(self.encoder.encoding_size, teacher_dim)

        # Weights initialization
        nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.predictor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.predictor.bias)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.masked_encoding_retention(x, mask, mask_token=self.mask_token, return_keys="features")
        features = latent["features"].flatten(2).permute(0, 2, 1)

        return self.predictor(features)

    def forward_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(torch.bool)
        pred_masked = pred[mask]
        target_masked = target[mask]
        loss = F.cosine_similarity(pred_masked.float(), target_masked.float(), dim=-1)  # pylint: disable=not-callable

        return -loss.mean()

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        pred = self.forward_features(x, mask)
        loss = self.forward_loss(pred, target_tokens, mask)

        return {"loss": loss, "pred": pred, "mask": mask}
