"""
data2vec, adapted from
https://github.com/arxyzan/data2vec-pytorch/blob/main/data2vec/data2vec.py
and
https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec_vision.py

Paper "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language",
https://arxiv.org/abs/2202.03555

Changes from original:
* The head (final_proj in original), not applied in the vision modality - applied here
"""

# Reference license: MIT (both)

import copy
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet


class Data2Vec(SSLBaseNet):
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
        assert isinstance(self.backbone, MaskedTokenRetentionMixin)

        normalize_targets: bool = self.config["normalize_targets"]
        average_top_k_layers: int = self.config["average_top_k_layers"]
        loss_beta: float = self.config["loss_beta"]

        self.normalize_targets = normalize_targets
        self.average_top_k_layers = average_top_k_layers
        self.loss_beta = loss_beta

        self.ema_backbone = copy.deepcopy(self.backbone)
        self.head = nn.Linear(self.backbone.embedding_size, self.backbone.embedding_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width), requires_grad=True)

        # Weights initialization
        self.ema_backbone.load_state_dict(self.backbone.state_dict())

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, src: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.backbone.masked_encoding_retention(src, mask, mask_token=self.mask_token)["features"]
        x = x.flatten(2).transpose(1, 2)

        y = self.ema_backbone.masked_encoding_omission(src, return_all_features=True)["tokens"]
        y = y[:, self.ema_backbone.num_special_tokens :]
        y = y[..., -self.average_top_k_layers :]  # Take the last k layers
        y = y.permute(3, 0, 1, 2)

        y = [F.layer_norm(t.float(), t.shape[-1:]) for t in y[:-1]] + [y[-1]]
        y = sum(y) / len(y)
        if self.normalize_targets is True:
            y = F.layer_norm(y.float(), y.shape[-1:])

        mask = ~mask.bool()
        x = x[mask]
        y = y[mask]

        x = self.head(x)
        loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta).sum(dim=-1).mean()

        return loss
