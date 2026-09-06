"""
NEPA (Next-Embedding Predictive Autoregression), adapted from
https://github.com/SihanXU/nepa/blob/main/models/vit_nepa/modeling_vit_nepa.py

Paper "Next-Embedding Prediction Makes Strong Vision Learners",
https://arxiv.org/abs/2512.16922
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F

from birder.layers.moe import MoETrainingOutputType
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet


def prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, shift: bool = True, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if shift is True:
        pred = pred[:, :-1, :]
        target = target[:, 1:, :]
        if valid_mask is not None:
            valid_mask = valid_mask[:, :-1] & valid_mask[:, 1:]

    target = target.detach()

    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)

    loss = -(pred * target).sum(dim=-1)
    if valid_mask is None:
        return loss.mean()

    valid_mask = valid_mask.to(dtype=loss.dtype)
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


class NEPA(SSLBaseNet):
    def __init__(
        self,
        backbone: PreTrainEncoder,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"
        assert isinstance(self.backbone, MaskedTokenOmissionMixin)

        self.shift: bool = self.config.get("shift", True)
        self.remove_reg_tokens: bool = self.config.get("remove_reg_tokens", False)

        if hasattr(self.backbone, "set_causal_attention") is False:
            raise ValueError("NEPA requires a backbone with set_causal_attention support")

        self.backbone.set_causal_attention(True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_moe_training_output: bool = False,
    ) -> dict[str, Any]:
        moe_training_output: Optional[MoETrainingOutputType] = None
        if grid_sizes is None:
            if return_moe_training_output is True:
                features, moe_training_output = self.backbone.forward_features(  # type: ignore[call-arg]
                    x, return_input_embedding=True, return_moe_training_output=True
                )
            else:
                features = self.backbone.forward_features(x, return_input_embedding=True)  # type: ignore[call-arg]
        else:
            if return_moe_training_output is True:
                features, moe_training_output = self.backbone.forward_features(  # type: ignore[call-arg]
                    x,
                    return_input_embedding=True,
                    grid_sizes=grid_sizes,
                    valid_mask=valid_mask,
                    return_moe_training_output=True,
                )
            else:
                features = self.backbone.forward_features(  # type: ignore[call-arg]
                    x, return_input_embedding=True, grid_sizes=grid_sizes, valid_mask=valid_mask
                )

        prediction_valid_mask: Optional[torch.Tensor] = None
        if valid_mask is not None:
            num_special_tokens = getattr(self.backbone, "num_special_tokens", 0)
            special_valid_mask = valid_mask.new_ones((valid_mask.size(0), num_special_tokens))
            prediction_valid_mask = torch.concat([special_valid_mask, valid_mask], dim=1)

        if self.remove_reg_tokens is True:
            # Strip register tokens
            num_reg = getattr(self.backbone, "num_reg_tokens", 0)
            features = features[:, num_reg:, :, :]
            if prediction_valid_mask is not None:
                prediction_valid_mask = prediction_valid_mask[:, num_reg:]

        target, pred = features.unbind(dim=-1)
        loss = prediction_loss(pred, target, shift=self.shift, valid_mask=prediction_valid_mask)

        result = {"loss": loss}
        if moe_training_output is not None:
            result["moe_training_output"] = moe_training_output

        return result
