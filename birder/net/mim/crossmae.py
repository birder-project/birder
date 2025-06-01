"""
CrossMAE, adapted from
https://github.com/TonyLianLong/CrossMAE/blob/main/models_cross.py

Paper "Rethinking Patch Dependence for Masked Autoencoders",
https://arxiv.org/abs/2401.14391
"""

# Reference license: Attribution-NonCommercial 4.0 International

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP

from birder.common.masking import uniform_mask
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.mim.base import MIMBaseNet


class WeightedFeatureMaps(nn.Module):
    def __init__(self, k: int, decoder_depth: int) -> None:
        super().__init__()
        self.linear = nn.Linear(k, decoder_depth, bias=False)

    def forward(self, stacked_feature_maps: torch.Tensor) -> torch.Tensor:
        output = self.linear(stacked_feature_maps)

        return output


class CrossAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(decoder_dim, decoder_dim)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2)
        self.proj = nn.Linear(decoder_dim, decoder_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        (B, N, C) = tgt.size()
        n_kv = memory.size(1)
        q = self.q(tgt).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(B, n_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (k, v) = kv.unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # pylint: disable=not-callable
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.cross_attn = CrossAttention(encoder_dim, decoder_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.mlp = MLP(decoder_dim, [int(decoder_dim * mlp_ratio), decoder_dim], activation_layer=nn.GELU)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = tgt + self.cross_attn(self.norm1(tgt), memory)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossMAE(MIMBaseNet):
    default_size = (224, 224)

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
        assert isinstance(self.encoder, MaskedTokenOmissionMixin)

        self.mask_ratio = 0.75
        self.kept_mask_ratio = 0.25
        self.patch_size = self.encoder.stem_stride
        encoder_dim = self.encoder.embedding_size
        decoder_embed_dim = 512
        decoder_depth = 8

        self.wfm = WeightedFeatureMaps(self.encoder.num_layers, decoder_depth=decoder_depth)
        self.decoder_norms = nn.ModuleList()
        for _ in range(decoder_depth - 1):
            self.decoder_norms.append(nn.LayerNorm(encoder_dim, eps=1e-6))

        self.decoder_norms.append(nn.Identity())  # Last norm uses the encoder native norm

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sin-cos embedding
        pos_embedding = pos_embedding_sin_cos_2d(
            h=self.size[0] // self.patch_size,
            w=self.size[1] // self.patch_size,
            dim=decoder_embed_dim,
            num_special_tokens=0,
        ).unsqueeze(0)
        self.decoder_pos_embed = nn.Parameter(pos_embedding, requires_grad=False)

        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_depth):
            self.decoder_layers.append(CrossAttentionBlock(encoder_dim, decoder_embed_dim, num_heads=16, mlp_ratio=4.0))

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * self.encoder.input_channels, bias=True)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """

        p = self.patch_size
        h = int(x.shape[1] ** 0.5)
        w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def mask_tokens_grid(self, mask: torch.Tensor) -> torch.Tensor:
        N = mask.size(0)
        x = self.decoder_pos_embed.masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.size(-1))
        x = x + self.mask_token

        return x

    def forward_decoder(self, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.mask_tokens_grid(mask)
        memory = self.wfm(memory)

        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, self.decoder_norms[i](memory[..., i]))

        x = self.decoder_norm(x)
        x = self.pred(x)

        return x

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(x)

        target = target.masked_select(mask.bool().unsqueeze(-1)).reshape(target.size(0), -1, target.size(-1))

        # Normalize pixels per-patch
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean()

        return loss

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = (self.size[0] // self.encoder.stem_stride) * (self.size[1] // self.encoder.stem_stride)
        (mask, ids_keep, _) = uniform_mask(x.size(0), seq_len, self.mask_ratio, self.kept_mask_ratio, device=x.device)

        latent = self.encoder.masked_encoding_omission(x, ids_keep, return_all_features=True)["tokens"]
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(x, pred, mask)

        return {"loss": loss, "pred": pred, "mask": mask}
