"""
RoPE ViT, adapted from
https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_sincos.py

Paper "Rotary Position Embedding for Vision Transformer", https://arxiv.org/abs/2403.13298

Changes from original:
* Implemented only axial RoPE (EVA style RoPE)
* Modified rotate_half (original implementation seems off)
"""

# Reference license: Apache-2.0 and Apache-2.0

import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.vit import LayerScale
from birder.net.vit import MultiHeadAttentionPool
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding


def build_rotary_pos_embed(
    dim: int, temperature: float, grid_size: tuple[int, int], pt_grid_size: Optional[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 4 == 0
    num_bands = dim // 4
    exp = torch.arange(0, num_bands, 1) / num_bands
    bands = 1.0 / (temperature**exp)

    if pt_grid_size is None:
        pt_grid_size = grid_size

    t = [torch.arange(s) / s * p for s, p in zip(grid_size, pt_grid_size)]
    grid = torch.stack(torch.meshgrid(t, indexing="ij"), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands
    sin_emb = pos.sin()
    cos_emb = pos.cos()

    num_spatial_dim = grid_size[0] * grid_size[1]

    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)

    return (sin_emb, cos_emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Taken from: https://github.com/facebookresearch/capi/blob/main/model.py
    (x1, x2) = x.chunk(2, dim=-1)
    return torch.concat((-x2, x1), dim=-1)


def apply_rotary_pos_embed(x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
    (sin_emb, cos_emb) = embed.tensor_split(2, dim=-1)
    if cos_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rotate_half(x) * sin_emb.unsqueeze(1).expand_as(x)

    return x * cos_emb + rotate_half(x) * sin_emb


class SequentialWithRope(nn.Sequential):
    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        for module in self:
            x = module(x, rope)

        return x


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        temperature: float,
        grid_size: tuple[int, int],
        pt_grid_size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        (sin_emb, cos_emb) = build_rotary_pos_embed(dim, temperature, grid_size=grid_size, pt_grid_size=pt_grid_size)
        self.pos_embed = nn.Buffer(torch.concat((sin_emb, cos_emb), dim=-1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_embed(x, self.pos_embed)


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float, num_special_tokens: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_special_tokens = num_special_tokens

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        n = self.num_special_tokens
        q = torch.concat([q[:, :, :n, :], apply_rotary_pos_embed(q[:, :, n:, :], rope)], dim=2)
        k = torch.concat([k[:, :, :n, :], apply_rotary_pos_embed(k[:, :, n:, :], rope)], dim=2)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: Optional[int],
        num_special_tokens: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        activation_layer: Callable[..., nn.Module],
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        super().__init__()

        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Attention block
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = RoPEAttention(
            hidden_dim, num_heads, attn_drop=attention_dropout, proj_drop=dropout, num_special_tokens=num_special_tokens
        )
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        # MLP block
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(
            hidden_dim, [mlp_dim, hidden_dim], activation_layer=activation_layer, inplace=None, dropout=dropout
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x), rope)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_special_tokens: int,
        dropout: float,
        attention_dropout: float,
        dpr: list[float],
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        layers = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        for i in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    num_special_tokens,
                    dropout,
                    attention_dropout,
                    dpr[i],
                    activation_layer=nn.GELU,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )

        self.block = SequentialWithRope(*layers)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        return self.block(x, rope)

    def forward_features(self, x: torch.Tensor, rope: torch.Tensor) -> list[torch.Tensor]:
        xs = []
        for blk in self.block:
            x = blk(x, rope)
            xs.append(x)

        return xs


class MAEDecoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        num_special_tokens: int,
        activation_layer: Callable[..., nn.Module],
        grid_size: tuple[int, int],
    ) -> None:
        super().__init__()
        mlp_dim = hidden_dim * 4
        self.rope = RoPE(hidden_dim // num_heads, temperature=100.0, grid_size=grid_size)

        # Attention block
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = RoPEAttention(
            hidden_dim, num_heads, attn_drop=0.0, proj_drop=0.0, num_special_tokens=num_special_tokens
        )

        # MLP block
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(hidden_dim, [mlp_dim, hidden_dim], activation_layer=activation_layer, inplace=None, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.rope.pos_embed)
        x = x + self.mlp(self.norm2(x))

        return x


# pylint: disable=invalid-name,too-many-instance-attributes
class RoPE_ViT(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin):
    block_group_regex = r"encoder\.block\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        image_size = self.size
        attention_dropout = 0.0
        dropout = 0.0
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        attn_pool_head: bool = self.config.get("attn_pool_head", False)
        pt_grid_size: Optional[tuple[int, int]] = self.config.get("pt_grid_size", None)
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_reg_tokens = num_reg_tokens
        self.rope_temperature = 100.0

        # Cast in case config was loaded from a json (no tuples),
        # TorchScript does not accept a list when tuple expected
        if isinstance(pt_grid_size, list):
            pt_grid_size = tuple(pt_grid_size)  # type: ignore[unreachable]

        self.pt_grid_size = pt_grid_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # Stochastic depth decay rule

        self.conv_proj = nn.Conv2d(
            self.input_channels,
            hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            bias=True,
        )
        self.patch_embed = PatchEmbed()

        seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.num_special_tokens = 0

        # Add a class token
        if class_token is True:
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1
            self.num_special_tokens += 1
        else:
            self.class_token = None

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
            seq_length += self.num_reg_tokens
            self.num_special_tokens += self.num_reg_tokens
        else:
            self.reg_tokens = None

        # Add positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

        # RoPE
        self.rope = RoPE(
            hidden_dim // num_heads,
            temperature=self.rope_temperature,
            grid_size=(image_size[0] // patch_size, image_size[1] // patch_size),
            pt_grid_size=self.pt_grid_size,
        )

        # Encoder
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            self.num_special_tokens,
            dropout,
            attention_dropout,
            dpr,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        if attn_pool_head is True:
            self.attn_pool = MultiHeadAttentionPool(hidden_dim, num_heads, mlp_dim, qkv_bias=True, latent_len=1)
        else:
            self.attn_pool = nn.Identity()

        self.return_stages = ["neck"]  # Actually meaningless, but for completeness
        self.return_channels = [hidden_dim]
        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = hidden_dim
        self.encoding_size = hidden_dim
        self.decoder_block = partial(
            MAEDecoderBlock,
            16,
            num_special_tokens=self.num_special_tokens,
            activation_layer=nn.GELU,
            grid_size=(image_size[0] // patch_size, image_size[1] // patch_size),
        )

        # Weight initialization
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.classifier, nn.Linear):
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.pos_embedding

        if H == self.size[0] and W == self.size[1]:
            return self.pos_embedding

        return adjust_position_embedding(
            self.pos_embedding,
            (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
            (H // self.patch_size, W // self.patch_size),
            self.num_special_tokens,
            antialias=False,
        )

    def _get_rope_embed(self, H: int, W: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.rope.pos_embed

        if H == self.size[0] and W == self.size[1]:
            return self.rope.pos_embed

        return torch.concat(
            build_rotary_pos_embed(
                self.hidden_dim // self.num_heads,
                self.rope_temperature,
                grid_size=(H // self.patch_size, W // self.patch_size),
                pt_grid_size=self.pt_grid_size,
            ),
            dim=-1,
        ).to(self.rope.pos_embed.device)

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

        if unfreeze_features is True:
            for param in self.attn_pool.parameters():
                param.requires_grad = True

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (H, W) = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        x = x + self._get_pos_embed(H, W)
        x = self.encoder(x, self._get_rope_embed(H, W))
        x = self.norm(x)

        x = x[:, self.num_special_tokens :]
        x = x.permute(0, 2, 1)
        (B, C, _) = x.size()
        x = x.reshape(B, C, self.size[0] // self.patch_size, self.size[1] // self.patch_size)

        return {self.return_stages[0]: x}

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.conv_proj.parameters():
            param.requires_grad = False

        self.pos_embedding.requires_grad = False

        for idx, module in enumerate(self.encoder.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def masked_encoding_omission(
        self, x: torch.Tensor, ids_keep: Optional[torch.Tensor] = None, return_all_features: bool = False
    ) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Add pos embedding without special tokens
        x = x + self.pos_embedding[:, self.num_special_tokens :, :]

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

            rope_dim = self.rope.pos_embed.size(1)
            rope = self.rope.pos_embed.unsqueeze(0).repeat(x.size(0), 1, 1)
            rope_masked = torch.gather(rope, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, rope_dim))
        else:
            rope_masked = self.rope.pos_embed

        # Append class and register tokens
        if self.class_token is not None:
            cls_token = self.class_token + self.pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
            batch_class_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.concat((batch_class_token, x), dim=1)

        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens + self.pos_embedding[:, 0 : self.num_reg_tokens, :]
            batch_reg_tokens = reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        # Apply transformer
        if return_all_features is True:
            xs = self.encoder.forward_features(x, rope_masked)
            xs[-1] = self.norm(xs[-1])
            x = torch.stack(xs, dim=-1)
        else:
            x = self.encoder(x, rope_masked)
            x = self.norm(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        (H, W) = x.shape[-2:]

        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        x = x + self._get_pos_embed(H, W)
        x = self.encoder(x, self._get_rope_embed(H, W))
        x = self.norm(x)
        x = self.attn_pool(x)

        if self.class_token is None:
            return x.mean(dim=1)

        # Classifier "token" as used by standard language architectures
        return x[:, self.num_reg_tokens]

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        self.dynamic_size = dynamic_size

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                self.num_special_tokens,
            )
        )

        # Adjust RoPE
        self.rope = RoPE(
            self.hidden_dim // self.num_heads,
            temperature=self.rope_temperature,
            grid_size=(new_size[0] // self.patch_size, new_size[1] // self.patch_size),
            pt_grid_size=self.pt_grid_size,
        )

        # Define adjusted decoder block
        self.decoder_block = partial(
            MAEDecoderBlock,
            16,
            num_special_tokens=self.num_special_tokens,
            activation_layer=nn.GELU,
            grid_size=(new_size[0] // self.patch_size, new_size[1] // self.patch_size),
        )


registry.register_alias(
    "rope_vit_s32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_s16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_s14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_m32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_m16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_m14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_b32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_b16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_b14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vit_l32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vit_l16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vit_l14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vit_h16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vit_h14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(  # From "Scaling Vision Transformers"
    "rope_vit_g14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 40,
        "num_heads": 16,
        "hidden_dim": 1408,
        "mlp_dim": 6144,
        "drop_path_rate": 0.1,
    },
)

# With registers
registry.register_alias(
    "rope_vitreg4_s32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_s16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_s14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_m32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_m16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_m14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_b32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_b16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_b14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_vitreg4_l32",
    RoPE_ViT,
    config={
        "patch_size": 32,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg4_l16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg4_l14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg4_h16",
    RoPE_ViT,
    config={
        "patch_size": 16,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg4_h14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(  # From "Scaling Vision Transformers"
    "rope_vitreg4_g14",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 40,
        "num_heads": 16,
        "hidden_dim": 1408,
        "mlp_dim": 6144,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)

# Shape-optimized vision transformer (SoViT)
registry.register_alias(
    "rope_vit_so150m_p14_ap",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vit_so400m_p14_ap",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 27,
        "num_heads": 16,
        "hidden_dim": 1152,
        "mlp_dim": 4304,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg4_so150m_p14_ap",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "num_reg_tokens": 4,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg8_so150m_p14_ap",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg4_so400m_p14_ap",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 27,
        "num_heads": 16,
        "hidden_dim": 1152,
        "mlp_dim": 4304,
        "num_reg_tokens": 4,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_vitreg8_so400m_p14_ap",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 27,
        "num_heads": 16,
        "hidden_dim": 1152,
        "mlp_dim": 4304,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
