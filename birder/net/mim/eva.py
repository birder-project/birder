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
        self.predictor = nn.Linear(self.encoder.feature_dim, teacher_dim)

        # Weights initialization
        nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.predictor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.predictor.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if grid_sizes is None:
            latent = self.encoder.masked_encoding_retention(x, mask, mask_token=self.mask_token, return_keys="features")
        else:
            latent = self.encoder.masked_encoding_retention(
                x,
                mask,
                mask_token=self.mask_token,
                return_keys="features",
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )

        features = latent["features"].flatten(2).permute(0, 2, 1)
        pred = self.predictor(features)

        moe_auxiliary_loss = None
        if "auxiliary_losses" in latent:
            moe_auxiliary_loss = latent["auxiliary_losses"]["auxiliary_loss"]

        return (pred, moe_auxiliary_loss)

    def forward_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_mask = mask.to(torch.bool)
        if valid_mask is not None:
            loss_mask = loss_mask & valid_mask

        pred_masked = pred[loss_mask]
        target_masked = target[loss_mask]
        loss = F.cosine_similarity(pred_masked.float(), target_masked.float(), dim=-1)  # pylint: disable=not-callable

        return -loss.mean()

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        pred, moe_auxiliary_loss = self.forward_features(x, mask, grid_sizes=grid_sizes, valid_mask=valid_mask)
        loss = self.forward_loss(pred, target_tokens, mask, valid_mask=valid_mask)

        result = {"loss": loss, "pred": pred, "mask": mask}
        if moe_auxiliary_loss is not None:
            result["moe_auxiliary_loss"] = moe_auxiliary_loss

        return result
