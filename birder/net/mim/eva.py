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

from birder.layers.moe import MoETrainingOutputType
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
        return_moe_training_output: bool = False,
    ) -> tuple[torch.Tensor, Optional[MoETrainingOutputType]]:
        if grid_sizes is None:
            if return_moe_training_output is True:
                latent = self.encoder.masked_encoding_retention(
                    x, mask, mask_token=self.mask_token, return_keys="features", return_moe_training_output=True
                )
            else:
                latent = self.encoder.masked_encoding_retention(
                    x, mask, mask_token=self.mask_token, return_keys="features"
                )
        else:
            if return_moe_training_output is True:
                latent = self.encoder.masked_encoding_retention(
                    x,
                    mask,
                    mask_token=self.mask_token,
                    return_keys="features",
                    grid_sizes=grid_sizes,
                    valid_mask=valid_mask,
                    return_moe_training_output=True,
                )
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

        moe_training_output = None
        if "moe_training_output" in latent:
            moe_training_output = latent["moe_training_output"]

        return (pred, moe_training_output)

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
        return_moe_training_output: bool = False,
    ) -> dict[str, Any]:
        # pylint: disable=arguments-differ

        pred, moe_training_output = self.forward_features(
            x, mask, grid_sizes=grid_sizes, valid_mask=valid_mask, return_moe_training_output=return_moe_training_output
        )
        loss = self.forward_loss(pred, target_tokens, mask, valid_mask=valid_mask)

        result = {"loss": loss, "pred": pred, "mask": mask}
        if moe_training_output is not None:
            result["moe_training_output"] = moe_training_output

        return result
