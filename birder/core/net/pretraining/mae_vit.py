"""
MAE ViT, adapted from
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
and
https://github.com/facebookresearch/mae/blob/main/models_mae.py

Paper "Masked Autoencoders Are Scalable Vision Learners",
https://arxiv.org/abs/2111.06377
"""

# Reference license: MIT and Attribution-NonCommercial 4.0 International

from typing import Optional

import torch
from torch import nn

from birder.core.net.base import PreTrainEncoder
from birder.core.net.pretraining.base import PreTrainBaseNet
from birder.core.net.vit import ViT


# pylint: disable=invalid-name
class MAE_ViT(PreTrainBaseNet):
    default_size = 224

    def __init__(
        self,
        encoder: PreTrainEncoder,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(encoder, net_param, size)
        assert self.net_param is None, "net-param not supported"
        assert isinstance(self.encoder, ViT) is True, "Only ViT is supported as an encoder for this network"
        self.encoder: ViT

        self.mask_ratio = 0.75
        num_patches = self.encoder.encoding_size // self.encoder.embedding_size  # Include special tokens
        self.patch_size = int(self.size / ((num_patches - 1 - self.encoder.num_reg_tokens) ** 0.5))
        encoder_dim = self.encoder.embedding_size
        decoder_embed_dim = 512
        decoder_depth = 8

        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim).normal_(std=0.02))

        layers = []
        for _ in range(decoder_depth):
            layers.append(self.encoder.decoder_block(decoder_embed_dim))

        layers.append(nn.LayerNorm(decoder_embed_dim, eps=1e-6))
        layers.append(
            nn.Linear(decoder_embed_dim, self.encoder.patch_size**2 * self.encoder.input_channels, bias=True)
        )  # Decoder to patch
        self.decoder = nn.Sequential(*layers)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """

        p = self.encoder.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """

        p = self.encoder.patch_size
        h = int(x.shape[1] ** 0.5)
        w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        special_token_len = 1 + self.encoder.num_reg_tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + special_token_len - x.shape[1], 1)
        x_ = torch.concat([x[:, special_token_len:, :], mask_tokens], dim=1)  # No special tokens
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Un-shuffle
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

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches

        return loss

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (latent, mask, ids_restore) = self.encoder.masked_encoding(x, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return {"loss": loss, "pred": pred, "mask": mask}
