"""
FCMAE, adapted from
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/fcmae.py

Paper "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders",
https://arxiv.org/abs/2301.00808
"""

# Reference license: Attribution-NonCommercial 4.0 International

from typing import Optional

import torch
from torch import nn

from birder.core.net.base import PreTrainEncoder
from birder.core.net.mim.base import MIMBaseNet


class FCMAE(MIMBaseNet):
    default_size = 224

    def __init__(
        self,
        encoder: PreTrainEncoder,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(encoder, net_param, size)
        assert self.net_param is None, "net-param not supported"

        self.mask_ratio = 0.6
        self.decoder_embed_dim = 512
        self.decoder_depth = 1
        self.patch_size = 32  # Hardcoded at ConvNeXt v2

        self.proj = nn.Conv2d(
            in_channels=self.encoder.encoding_size,
            out_channels=self.decoder_embed_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_embed_dim, 1, 1))

        layers = []
        for _ in range(self.decoder_depth):
            layers.append(self.encoder.decoder_block(self.decoder_embed_dim))

        self.decoder = nn.Sequential(*layers)

        self.pred = nn.Conv2d(
            in_channels=self.decoder_embed_dim,
            out_channels=self.patch_size**2 * self.encoder.input_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.LayerNorm) is True:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            if hasattr(self, "mask_token") is True:
                torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """

        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, conv_out**2, L*3) or (N, L*3, conv_out, conv_out)
        imgs: (N, 3, H, W)
        """

        if x.ndim == 4:
            (n, c, _, _) = x.shape
            x = x.reshape(n, c, -1)
            x = torch.einsum("ncl->nlc", x)

        p = self.patch_size
        h = int(x.shape[1] ** 0.5)
        w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def forward_decoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # Append mask token
        (n, _c, h, w) = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(n, 1, h, w)
        x = x * (1.0 - mask) + mask_token * mask

        # Decoding
        x = self.decoder(x)
        x = self.pred(x)

        return x

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: 0 is keep, 1 is remove
        """

        (n, c, _, _) = pred.shape
        pred = pred.reshape(n, c, -1)
        pred = torch.einsum("ncl->nlc", pred)

        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches

        return loss

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (latent, mask) = self.encoder.masked_encoding(x, self.mask_ratio)
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(x, pred, mask)
        return {"loss": loss, "pred": pred, "mask": mask}