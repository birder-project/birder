"""
data2vec 2.0, adapted from
https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py

Paper "Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language",
https://arxiv.org/abs/2212.07525

Changes from original:
* Target CLS is taken just from the last layer
* Replaced instance norm (1st of the IN -> AVG -> LM) with layer norm
"""

# Reference license: MIT

import copy
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.common.masking import get_ids_keep
from birder.layers import LayerNorm2d
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet


class Decoder2d(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int, num_layers: int, H: int, W: int) -> None:
        super().__init__()

        self.H = H
        self.W = W

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=16,
                ),
                LayerNorm2d(embed_dim),
                nn.GELU(),
            )
        )
        for _ in range(num_layers - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        embed_dim,
                        embed_dim,
                        kernel_size=(kernel_size, kernel_size),
                        stride=(1, 1),
                        padding=(1, 1),
                        groups=16,
                    ),
                    LayerNorm2d(embed_dim),
                    nn.GELU(),
                )
            )

        self.proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, C) = x.size()  # B, N, C

        x = x.transpose(1, 2).reshape(B, C, self.H, self.W)

        residual = x
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i > 0:
                x = x + residual

            residual = x

        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        return x


class Data2Vec2(SSLBaseNet):
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

        average_top_k_layers: int = self.config["average_top_k_layers"]
        decoder_dim: int = self.config["decoder_dim"]
        decoder_kernel_size: int = self.config["decoder_kernel_size"]
        decoder_layers: int = self.config["decoder_layers"]
        clone_batch: int = self.config["clone_batch"]
        cls_loss_weight: float = self.config["cls_loss_weight"]

        self.average_top_k_layers = average_top_k_layers
        self.clone_batch = clone_batch
        self.cls_loss_weight = cls_loss_weight
        self.num_patches = (self.size[0] // self.backbone.max_stride) * (self.size[1] // self.backbone.max_stride)

        self.ema_backbone = copy.deepcopy(self.backbone)
        self.decoder = Decoder2d(
            self.backbone.embedding_size,
            decoder_dim,
            decoder_kernel_size,
            decoder_layers,
            self.size[0] // self.backbone.max_stride,
            self.size[1] // self.backbone.max_stride,
        )
        self.head = nn.Linear(self.backbone.embedding_size, self.backbone.embedding_size)

        # Weights initialization
        self.ema_backbone.load_state_dict(self.backbone.state_dict())

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, src: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        # Target Representations
        y = self.ema_backbone.masked_encoding_omission(src, return_all_features=True, return_keys="all")
        y_cls = y["embedding"]
        y = y["tokens"][:, self.ema_backbone.num_special_tokens :]
        y = y[..., -self.average_top_k_layers :]  # Take the last k layers
        y = y.permute(3, 0, 1, 2)

        y = [F.layer_norm(t.float(), t.shape[-1:]) for t in y[:-1]] + [y[-1]]
        y = sum(y) / len(y)
        y = F.layer_norm(y.float(), y.shape[-1:])

        if self.clone_batch > 1:
            y = y.repeat_interleave(self.clone_batch, 0)
            y_cls = y_cls.repeat_interleave(self.clone_batch, 0)
            src = src.repeat_interleave(self.clone_batch, 0)

        ids_keep = get_ids_keep(masks)
        x = self.backbone.masked_encoding_omission(src, ids_keep=ids_keep, return_keys="all")
        x_cls = x["embedding"]
        x = x["tokens"][:, self.backbone.num_special_tokens :]

        x_cls = self.head(x_cls)

        # Using noise instead of mask tokens
        full_sequence = (
            torch.randn(x.size(0), self.num_patches, self.backbone.embedding_size, device=x.device, dtype=x.dtype)
            * 0.02
        )
        indices_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, self.backbone.embedding_size)  # (B, num_kept, d_model)
        full_sequence.scatter_(1, indices_expanded, x)
        masks = masks.bool()

        predictions = self.decoder(full_sequence)

        # Compute loss only on masked positions
        patch_loss = F.mse_loss(predictions[masks], y[masks], reduction="none").sum(dim=-1).mean()

        # CLS loss
        cls_loss = F.mse_loss(x_cls, y_cls, reduction="none").sum(dim=-1).mean() * self.cls_loss_weight

        return patch_loss + cls_loss
