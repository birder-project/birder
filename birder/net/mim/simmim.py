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

    def __init__(
        self,
        encoder: PreTrainEncoder,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(encoder, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"
        assert isinstance(self.encoder, MaskedTokenRetentionMixin)

        self.mask_ratio = 0.6
        self.patch_size = encoder.max_stride

        # self.mask_token = nn.Parameter(torch.zeros(1, self.encoder.embedding_size, 1, 1))
        self.decoder = nn.Conv2d(
            in_channels=self.encoder.embedding_size,
            out_channels=self.patch_size**2 * 3,
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

    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

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
        seq_len = (self.size[0] // self.encoder.max_stride) * (self.size[1] // self.encoder.max_stride)
        mask = uniform_mask(x.size(0), seq_len, mask_ratio=self.mask_ratio, device=x.device)[0]

        latent = self.encoder.masked_encoding_retention(x, mask, mask_token=self.mask_token)
        pred = self.forward_decoder(latent["features"])
        loss = self.forward_loss(x, pred, mask)

        return {"loss": loss, "pred": pred, "mask": mask}
