"""
MAE ViT, adapted from
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
and
https://github.com/facebookresearch/mae/blob/main/models_mae.py

Paper "Masked Autoencoders Are Scalable Vision Learners",
https://arxiv.org/abs/2111.06377
"""

# Reference license: MIT and Attribution-NonCommercial 4.0 International

from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.common.masking import uniform_mask
from birder.model_registry import registry
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.mim.base import MIMBaseNet


# pylint: disable=invalid-name
class MAE_ViT(MIMBaseNet):
    default_size = (224, 224)
    default_mask_ratio: float = 0.75
    auto_register = False

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
        assert isinstance(self.encoder, MaskedTokenOmissionMixin)
        assert hasattr(self.encoder, "decoder_block")

        self.patch_size = self.encoder.stem_stride
        encoder_dim = self.encoder.embedding_size
        decoder_embed_dim: int = self.config["decoder_embed_dim"]
        decoder_depth: int = self.config["decoder_depth"]
        norm_pix_loss: bool = self.config.get("norm_pix_loss", False)

        self.norm_pix_loss = norm_pix_loss

        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sin-cos embedding
        pos_embedding = pos_embedding_sin_cos_2d(
            h=self.size[0] // self.patch_size,
            w=self.size[1] // self.patch_size,
            dim=decoder_embed_dim,
            num_special_tokens=self.encoder.num_special_tokens,
        )
        self.decoder_pos_embed = nn.Parameter(pos_embedding, requires_grad=False)

        layers = []
        for _ in range(decoder_depth):
            layers.append(self.encoder.decoder_block(decoder_embed_dim))

        layers.append(nn.LayerNorm(decoder_embed_dim, eps=1e-6))
        layers.append(
            nn.Linear(decoder_embed_dim, self.patch_size**2 * self.input_channels, bias=True)
        )  # Decoder to patch
        self.decoder = nn.Sequential(*layers)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """

        p = self.patch_size
        c = self.input_channels
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """

        p = self.patch_size
        c = self.input_channels
        h = int(x.size(1) ** 0.5)
        w = int(x.size(1) ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.size(0), h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.size(0), c, h * p, w * p))

        return imgs

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        special_token_len = self.encoder.num_special_tokens
        mask_tokens = self.mask_token.repeat(x.size(0), ids_restore.size(1) + special_token_len - x.size(1), 1)
        x_ = torch.concat([x[:, special_token_len:, :], mask_tokens], dim=1)  # No special tokens
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.size(2)))  # Un-shuffle
        x = torch.concat([x[:, :special_token_len, :], x_], dim=1)  # Append special tokens

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply transformer
        x = self.decoder(x)

        # Remove special tokens
        x = x[:, special_token_len:, :]

        return x

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(x)

        if self.norm_pix_loss is True:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches

        return loss

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.size[0] // self.encoder.max_stride
        w = self.size[1] // self.encoder.max_stride
        (mask, ids_keep, ids_restore) = uniform_mask(
            x.size(0), h, w, self.mask_ratio, min_mask_size=self.min_mask_size, device=x.device
        )

        latent = self.encoder.masked_encoding_omission(x, ids_keep)["tokens"]
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)

        return {"loss": loss, "pred": pred, "mask": mask}


registry.register_model_config("mae_vit", MAE_ViT, config={"decoder_depth": 8, "decoder_embed_dim": 512})
registry.register_model_config("mae_vit_dec512d12", MAE_ViT, config={"decoder_depth": 12, "decoder_embed_dim": 512})
registry.register_model_config("mae_vit_dec512d24", MAE_ViT, config={"decoder_depth": 24, "decoder_embed_dim": 512})
registry.register_model_config("mae_vit_dec512d32", MAE_ViT, config={"decoder_depth": 32, "decoder_embed_dim": 512})
registry.register_model_config(
    "mae_vit_dec512d24_npl", MAE_ViT, config={"decoder_depth": 24, "decoder_embed_dim": 512, "norm_pix_loss": True}
)
registry.register_model_config(
    "mae_vit_dec512d32_npl", MAE_ViT, config={"decoder_depth": 32, "decoder_embed_dim": 512, "norm_pix_loss": True}
)
