"""
CAPI-DINO

Combines CAPI sparse masked patch prediction with a DINO-style global self-distillation head.
"""

from typing import Any
from typing import Optional

import torch

from birder.common import masking
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.capi import CAPIStudent
from birder.net.ssl.capi import CAPITeacher
from birder.net.ssl.dino_v2 import DINOHead


class CAPI_DINOStudent(CAPIStudent):
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

        dino_out_dim: int = self.config["dino_out_dim"]
        use_bn: bool = self.config["use_bn"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        head_bottleneck_dim: int = self.config["head_bottleneck_dim"]

        self.dino_head = DINOHead(
            self.backbone.embedding_size,
            dino_out_dim,
            use_bn=use_bn,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
        )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, ids_keep: torch.Tensor, ids_predict: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone.masked_encoding_omission(x, ids_keep, return_keys="all")
        tokens = out["tokens"]
        embedding = out["embedding"]

        mask = masking.mask_from_indices(ids_predict, self.seq_len)
        patch_logits = self.decoder(tokens, mask)
        patch_logits = self.head(patch_logits.flatten(0, 1))

        global_logits = self.dino_head(embedding)

        return (patch_logits, global_logits)


class CAPI_DINOTeacher(CAPITeacher):
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

        dino_out_dim: int = self.config["dino_out_dim"]
        use_bn: bool = self.config["use_bn"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        head_bottleneck_dim: int = self.config["head_bottleneck_dim"]

        self.dino_head = DINOHead(
            self.backbone.embedding_size,
            dino_out_dim,
            use_bn=use_bn,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
        )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, ids_keep: Optional[torch.Tensor], ids_predict: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.size(0)

        with torch.no_grad():
            out = self.backbone.masked_encoding_omission(x, ids_keep, return_keys="all")
            tokens = out["tokens"]
            embedding = out["embedding"]
            global_logits = self.dino_head(embedding)

        patch_tokens = tokens[:, self.backbone.num_special_tokens :, :]
        assignments, clustering_loss = self.head(patch_tokens.transpose(0, 1))

        assignments = assignments.detach().transpose(0, 1)
        row_indices = torch.arange(B, device=ids_predict.device).unsqueeze(1).expand_as(ids_predict)
        selected_assignments = assignments[row_indices, ids_predict]
        selected_assignments = selected_assignments.flatten(0, 1)

        return (selected_assignments, clustering_loss, global_logits)
