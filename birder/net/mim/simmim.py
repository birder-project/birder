"""
SimMIM, adapted from
https://github.com/microsoft/Swin-Transformer/blob/main/models/simmim.py

Paper "SimMIM: A Simple Framework for Masked Image Modeling",
https://arxiv.org/abs/2111.09886
"""

# Reference license: MIT

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.common.masking import uniform_mask
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.mim.base import MIMBaseNet


def norm_targets(targets: torch.Tensor, patch_size: int) -> torch.Tensor:
    assert patch_size % 2 == 1

    targets_count = torch.ones_like(targets)
    targets_square = targets**2.0

    targets_mean = F.avg_pool2d(  # pylint: disable=not-callable
        targets,
        kernel_size=(patch_size, patch_size),
        stride=(1, 1),
        padding=(patch_size // 2, patch_size // 2),
        count_include_pad=False,
    )
    targets_square_mean = F.avg_pool2d(  # pylint: disable=not-callable
        targets_square,
        kernel_size=(patch_size, patch_size),
        stride=(1, 1),
        padding=(patch_size // 2, patch_size // 2),
        count_include_pad=False,
    )
    targets_count = F.avg_pool2d(  # pylint: disable=not-callable
        targets_count,
        kernel_size=(patch_size, patch_size),
        stride=(1, 1),
        padding=(patch_size // 2, patch_size // 2),
        count_include_pad=True,
    ) * (patch_size**2)

    targets_var = (targets_square_mean - targets_mean**2.0) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.0)

    return (targets - targets_mean) / (targets_var + 1.0e-6) ** 0.5


class SimMIM(MIMBaseNet):
    default_size = (192, 192)
    default_mask_ratio = 0.6

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
        assert self.config is None, "config not supported"
        assert isinstance(self.encoder, MaskedTokenRetentionMixin)

        self.patch_size = encoder.max_stride

        self.decoder = nn.Conv2d(
            in_channels=self.encoder.embedding_size,
            out_channels=self.patch_size**2 * self.input_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.encoder.stem_width), requires_grad=True)

        # Weights initialization
        nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """

        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.input_channels, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.input_channels))

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, conv_out**2, L*C) or (N, L*C, conv_out, conv_out)
        imgs: (N, C, H, W)
        """

        if x.ndim == 4:
            (n, c, _, _) = x.shape
            x = x.reshape(n, c, -1)
            x = torch.einsum("ncl->nlc", x)

        p = self.patch_size
        h = int(x.shape[1] ** 0.5)
        w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.input_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.input_channels, h * p, w * p))

        return imgs

    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: 0 is keep, 1 is remove
        """

        (N, C, _, _) = pred.shape
        pred = pred.reshape(N, C, -1)
        pred = torch.einsum("ncl->nlc", pred)

        target = norm_targets(x, 47)
        target = self.patchify(target)

        loss = F.l1_loss(pred, target, reduction="none")
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches

        return loss

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.size[0] // self.encoder.max_stride
        w = self.size[1] // self.encoder.max_stride
        mask = uniform_mask(x.size(0), h, w, self.mask_ratio, min_mask_size=self.min_mask_size, device=x.device)[0]

        latent = self.encoder.masked_encoding_retention(x, mask, mask_token=self.mask_token)
        pred = self.forward_decoder(latent["features"])
        loss = self.forward_loss(x, pred, mask)

        return {"loss": loss, "pred": pred, "mask": mask}
