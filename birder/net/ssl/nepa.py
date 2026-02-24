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

from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, shift: bool = True) -> torch.Tensor:
    if shift is True:
        pred = pred[:, :-1, :]
        target = target[:, 1:, :]

    target = target.detach()

    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    return -(pred * target).sum(dim=-1).mean()


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

        if hasattr(self.backbone, "set_causal_attention") is False:
            raise ValueError("NEPA requires a backbone with set_causal_attention support")

        self.backbone.set_causal_attention(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x, return_input_embedding=True)  # type: ignore[call-arg]

        # Strip register tokens
        num_reg = getattr(self.backbone, "num_reg_tokens", 0)
        features = features[:, num_reg:, :, :]

        target, pred = features.unbind(dim=-1)

        return prediction_loss(pred, target, shift=self.shift)
