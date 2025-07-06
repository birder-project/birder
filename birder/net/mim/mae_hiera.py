"""
MAE Hiera, adapted from
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_mae.py

Paper "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles", https://arxiv.org/abs/2306.00989
"""

# Reference license: Apache-2.0

import math
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.common.masking import uniform_mask
from birder.model_registry import registry
from birder.net.hiera import Hiera
from birder.net.hiera import HieraBlock
from birder.net.hiera import undo_windowing
from birder.net.mim.base import MIMBaseNet


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    (B, num_mask_units) = x.shape[0:2]

    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])

    return x


# pylint: disable=invalid-name
class MAE_Hiera(MIMBaseNet):
    default_size = (224, 224)

    def __init__(
        self,
        encoder: Hiera,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(encoder, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"
        assert isinstance(self.encoder, Hiera)

        self.mask_ratio = 0.6
        encoder_dim = self.encoder.encoding_size
        decoder_embed_dim = 512
        decoder_depth = 8
        decoder_num_heads = 16
        self.patch_size = self.encoder.mask_unit_size[0] * self.encoder.patch_stride[0]

        self.mask_unit_spatial_shape_final = [
            i // s ** (self.encoder.q_pool) for i, s in zip(self.encoder.mask_unit_size, self.encoder.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.encoder.q_pool) for i, s in zip(self.encoder.tokens_spatial_shape, self.encoder.q_stride)
        ]

        self.encoder_norm = nn.LayerNorm(encoder_dim)

        curr_mu_size = list(self.encoder.mask_unit_size)
        self.multi_scale_fusion_heads = nn.ModuleList()
        stage_idx = 0
        for _ in self.encoder.stage_ends[: self.encoder.q_pool]:  # resolution constant after q_pool
            kernel = [i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.encoder.q_stride)]
            self.multi_scale_fusion_heads.append(
                nn.Conv2d(
                    self.encoder.return_channels[stage_idx],
                    encoder_dim,
                    kernel_size=kernel,
                    stride=kernel,
                    padding=(0, 0),
                )
            )
            stage_idx += 1

        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # MAE decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim)
        )

        layers = []
        for _ in range(decoder_depth):
            layers.append(
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    drop_path=0.0,
                    q_stride=1,
                    window_size=0,
                    use_mask_unit_attn=False,
                )
            )

        self.decoder_blocks = nn.Sequential(*layers)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Predictor
        self.pred_stride = self.encoder.patch_stride[-1] * (self.encoder.q_stride[-1] ** self.encoder.q_pool)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.encoder.q_stride))) * self.encoder.input_channels,
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.zeros_(m.bias)

    def get_pixel_label_2d(self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool) -> torch.Tensor:
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm is True:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """

        p = self.pred_stride
        h = int(x.shape[1] ** 0.5)
        w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.encoder.mask_spatial_shape)

        (mask, _, _) = uniform_mask(x.size(0), num_windows, self.mask_ratio, device=x.device)

        # Get multi-scale representations from encoder
        (intermediates, mask) = self.encoder.masked_encoding(x, mask)

        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.encoder.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, inter_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, inter_x)

        x = self.encoder_norm(x)

        return (x, mask)

    def forward_decoder(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.decoder_embed(x)

        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view((1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,))
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,  # type:ignore[arg-type]
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,  # type:ignore[arg-type]
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return (x, mask)

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        label = self.get_pixel_label_2d(x, mask, norm=True)

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (latent, mask) = self.forward_encoder(x)
        (pred, pred_mask) = self.forward_decoder(latent, mask)
        loss = self.forward_loss(x, pred, ~pred_mask)

        return {"loss": loss, "pred": pred, "mask": mask}


registry.register_weights(
    "mae_hiera_hiera_abswin_base",
    {
        "url": "https://huggingface.co/birder-project/mae_hiera_hiera_abswin_base/resolve/main",
        "description": "Masked auto-encoder Hiera with a Hiera abswin base image encoder, trained on 12M images",
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 328.1,
                "sha256": "bdf8a8495c99f4cb0f345cfcc5d616a427b59d46be0c5a7936766b60cd1efa60",
            }
        },
        "net": {"network": "mae_hiera"},
        "encoder": {"network": "hiera_abswin_base"},
    },
)
