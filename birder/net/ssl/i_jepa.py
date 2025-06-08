"""
I-JEPA (Image-based Joint-Embedding Predictive Architecture), adapted from
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py

Paper "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
https://arxiv.org/abs/2301.08243

Changes from original:
* Removed weight initialization with scaling per layer index
"""

# Reference license: Attribution-NonCommercial 4.0 International

import math
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.ssl.base import SSLBaseNet
from birder.net.vit import Encoder


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))

    return torch.concat(all_x, dim=0)


def repeat_interleave_batch(x: torch.Tensor, b: int, repeat: int) -> torch.Tensor:
    N = len(x) // b
    x = torch.concat([torch.concat([x[i * b : (i + 1) * b] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)
    return x


class MultiBlockMasking:
    # Adapted from: https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py

    def __init__(
        self,
        input_size: tuple[int, int],
        enc_mask_scale: tuple[float, float],
        pred_mask_scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        n_enc: int,
        n_pred: int,
        min_keep: int,
        allow_overlap: bool,
    ):
        self.height = input_size[0]
        self.width = input_size[1]
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.n_enc = n_enc
        self.n_pred = n_pred
        self.min_keep = min_keep  # Minimum number of patches to keep
        self.allow_overlap = allow_overlap  # Whether to allow overlap between enc and pred masks

    def _sample_block_size(
        self, scale: tuple[float, float], aspect_ratio_scale: tuple[float, float]
    ) -> tuple[int, int]:
        _rand = torch.rand(1).item()

        (min_s, max_s) = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        (min_ar, max_ar) = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(
        self, b_size: tuple[int, int], acceptable_regions: Optional[list[torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = b_size

        def constrain_mask(mask: torch.Tensor, tries: int = 0) -> None:
            """
            Helper to restrict given mask to a set of acceptable regions
            """

            assert acceptable_regions is not None
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # Loop to sample masks until we find a valid one
        tries = 0
        timeout = 20
        og_timeout = 20
        valid_mask = False
        while valid_mask is False:
            # Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1

            # Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)

            mask = torch.nonzero(mask.flatten())

            # If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if valid_mask is False:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout

        mask = mask.squeeze()
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0

        return (mask, mask_complement)

    def __call__(self, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Create encoder and predictor masks
        1. sample enc block (size + location)
        2. sample pred block (size)
        3. sample several enc block locations for each image
        4. sample several pred block locations for each image
        5. return enc mask and pred mask
        """

        p_size = self._sample_block_size(scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(scale=self.enc_mask_scale, aspect_ratio_scale=(1.0, 1.0))

        collated_masks_pred = []
        collated_masks_enc = []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(batch_size):
            masks_p = []
            masks_c = []
            for _ in range(self.n_pred):
                (mask, mask_c) = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_c.append(mask_c)
                min_keep_pred = min(min_keep_pred, len(mask))

            collated_masks_pred.append(masks_p)

            acceptable_regions: Optional[list[torch.Tensor]] = masks_c
            if self.allow_overlap is True:
                acceptable_regions = None

            masks_e = []
            for _ in range(self.n_enc):
                (mask, _) = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))

            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return (collated_masks_enc, collated_masks_pred)


class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        size: tuple[int, int],
        embed_dim: int,
        predictor_embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        depth: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        pos_embedding = pos_embedding_sin_cos_2d(h=size[0], w=size[1], dim=predictor_embed_dim, num_special_tokens=0)
        self.pos_embedding = nn.Parameter(pos_embedding, requires_grad=False)

        self.encoder = Encoder(
            depth, num_heads, predictor_embed_dim, mlp_dim, dropout=0.0, attention_dropout=0.0, dpr=dpr
        )
        self.norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, masks_encoder: list[torch.Tensor], masks_pred: list[torch.Tensor]
    ) -> torch.Tensor:
        B = len(x) // len(masks_encoder)

        # Project to predictor dim
        x = self.predictor_embed(x)

        # Apply positional embedding
        x_pos_embed = self.pos_embedding.repeat(B, 1, 1)
        x = x + apply_masks(x_pos_embed, masks_encoder)

        # Add positional embedding to mask tokens
        ctx = x.size(1)
        pos_embed = self.pos_embedding.repeat(B, 1, 1)
        pos_embed = apply_masks(pos_embed, masks_pred)
        pos_embed = repeat_interleave_batch(pos_embed, B, repeat=len(masks_encoder))

        pred_tokens = self.mask_token.repeat(pos_embed.size(0), pos_embed.size(1), 1)
        pred_tokens = pred_tokens + pos_embed

        # Concat mask tokens
        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.concat([x, pred_tokens], dim=1)

        # Encoder forward
        x = self.encoder(x)
        x = self.norm(x)

        # Mask tokens predictions
        x = x[:, ctx:]
        x = self.predictor_proj(x)

        return x


# pylint: disable=invalid-name
class I_JEPA(SSLBaseNet):
    def __init__(
        self,
        backbone: PreTrainEncoder,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is None, "config not supported"
        assert isinstance(self.backbone, MaskedTokenOmissionMixin)

    def forward(self, x: torch.Tensor, ids_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone.masked_encoding_omission(x, ids_keep)["tokens"]
