"""
Paper "ViT-5: Vision Transformers for The Mid-2020s", https://arxiv.org/abs/2602.08071
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import FFN
from birder.layers import LayerScale
from birder.layers import MultiHeadAttentionPool
from birder.layers import SwiGLU_FFN
from birder.layers.activations import get_activation_module
from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import GIANT
from birder.net._vit_configs import HUGE
from birder.net._vit_configs import LARGE
from birder.net._vit_configs import MEDIUM
from birder.net._vit_configs import SMALL
from birder.net._vit_configs import SO150
from birder.net._vit_configs import SO400
from birder.net._vit_configs import TINY
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import TokenRetentionResultType
from birder.net.base import normalize_out_indices
from birder.net.rope_vit import RoPE
from birder.net.rope_vit import apply_interleaved_rotary_pos_embed
from birder.net.rope_vit import apply_rotary_pos_embed
from birder.net.rope_vit import build_rotary_pos_embed
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding

RoPEPosEmbedType = tuple[torch.Tensor, Optional[torch.Tensor]]


def _register_rope_grid(num_reg_tokens: int) -> tuple[int, int]:
    reg_grid_size = math.isqrt(num_reg_tokens)
    if reg_grid_size * reg_grid_size == num_reg_tokens:
        return (reg_grid_size, reg_grid_size)

    return (num_reg_tokens, 1)


class SequentialWithRope(nn.Sequential):
    def forward(self, x: torch.Tensor, rope: RoPEPosEmbedType) -> torch.Tensor:  # pylint: disable=arguments-differ
        for module in self:
            x = module(x, rope)

        return x


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        num_special_tokens: int,
        num_reg_tokens: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
        rope_rot_type: str = "standard",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.is_causal = False
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_special_tokens = num_special_tokens
        self.num_reg_tokens = num_reg_tokens
        if rope_rot_type == "standard":
            self.apply_rot_fn = apply_rotary_pos_embed
        elif rope_rot_type == "interleaved":
            self.apply_rot_fn = apply_interleaved_rotary_pos_embed
        else:
            raise ValueError(f"Unknown rope_rot_type, got '{rope_rot_type}'")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm is True:
            self.q_norm = norm_layer(self.head_dim, eps=norm_layer_eps)
            self.k_norm = norm_layer(self.head_dim, eps=norm_layer_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_split_rope(
        self, x: torch.Tensor, patch_rope: torch.Tensor, reg_rope: Optional[torch.Tensor]
    ) -> torch.Tensor:
        segments = []
        if self.num_reg_tokens > 0 and reg_rope is not None:
            segments.append(self.apply_rot_fn(x[:, :, : self.num_reg_tokens, :], reg_rope))

        if self.num_special_tokens > self.num_reg_tokens:
            segments.append(x[:, :, self.num_reg_tokens : self.num_special_tokens, :])

        segments.append(self.apply_rot_fn(x[:, :, self.num_special_tokens :, :], patch_rope))

        return torch.concat(segments, dim=2)

    def forward(self, x: torch.Tensor, rope: RoPEPosEmbedType) -> torch.Tensor:
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        patch_rope, reg_rope = rope
        q = self._apply_split_rope(q, patch_rope, reg_rope)
        k = self._apply_split_rope(k, patch_rope, reg_rope)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=self.is_causal, scale=self.scale
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
        num_reg_tokens: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        activation_layer: Callable[..., nn.Module],
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        rope_rot_type: str = "standard",
    ) -> None:
        super().__init__()

        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Attention block
        self.norm1 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.attn = RoPEAttention(
            hidden_dim,
            num_heads,
            attn_drop=attention_dropout,
            proj_drop=dropout,
            num_special_tokens=num_special_tokens,
            num_reg_tokens=num_reg_tokens,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
            rope_rot_type=rope_rot_type,
        )
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        # MLP block
        self.norm2 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=dropout)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor, rope: RoPEPosEmbedType) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x), rope)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.attn.is_causal = is_causal


class Encoder(nn.Module):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_special_tokens: int,
        num_reg_tokens: int,
        dropout: float,
        attention_dropout: float,
        dpr: list[float],
        pre_norm: bool = False,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
        rope_rot_type: str = "standard",
    ) -> None:
        super().__init__()
        pre_layers = []
        if dropout > 0.0:
            pre_layers.append(nn.Dropout(dropout))
        if pre_norm is True:
            pre_layers.append(norm_layer(hidden_dim, eps=norm_layer_eps))

        self.pre_block = nn.Sequential(*pre_layers)

        layers = []
        for i in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    num_special_tokens,
                    num_reg_tokens,
                    dropout,
                    attention_dropout,
                    dpr[i],
                    activation_layer=activation_layer,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm_layer,
                    norm_layer_eps=norm_layer_eps,
                    mlp_layer=mlp_layer,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    rope_rot_type=rope_rot_type,
                )
            )

        self.block = SequentialWithRope(*layers)

    def forward(self, x: torch.Tensor, rope: RoPEPosEmbedType) -> torch.Tensor:
        x = self.pre_block(x)
        return self.block(x, rope)

    def forward_features(
        self, x: torch.Tensor, rope: RoPEPosEmbedType, out_indices: Optional[list[int]] = None
    ) -> list[torch.Tensor]:
        x = self.pre_block(x)

        out_indices_set = set(out_indices) if out_indices is not None else None
        xs = []
        for idx, blk in enumerate(self.block):
            x = blk(x, rope)
            if out_indices_set is None or idx in out_indices_set:
                xs.append(x)

        return xs

    def set_causal_attention(self, is_causal: bool = True) -> None:
        for b in self.block:
            b.set_causal_attention(is_causal)


class MAEDecoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        num_special_tokens: int,
        num_reg_tokens: int,
        activation_layer: Callable[..., nn.Module],
        grid_size: tuple[int, int],
        rope_grid_indexing: Literal["ij", "xy"],
        rope_grid_offset: int,
        rope_temperature: float,
        rope_reg_temperature: float,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
        rope_rot_type: str = "standard",
    ) -> None:
        super().__init__()
        mlp_dim = hidden_dim * 4
        self.rope = RoPE(
            hidden_dim // num_heads,
            temperature=rope_temperature,
            grid_size=grid_size,
            grid_indexing=rope_grid_indexing,
            grid_offset=rope_grid_offset,
            rope_rot_type=rope_rot_type,
        )
        if num_reg_tokens == 0:
            self.rope_reg = None
        else:
            self.rope_reg = RoPE(
                hidden_dim // num_heads,
                temperature=rope_reg_temperature,
                grid_size=_register_rope_grid(num_reg_tokens),
                grid_indexing=rope_grid_indexing,
                grid_offset=0,
                rope_rot_type=rope_rot_type,
            )

        # Attention block
        self.norm1 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.attn = RoPEAttention(
            hidden_dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            num_special_tokens=num_special_tokens,
            num_reg_tokens=num_reg_tokens,
            rope_rot_type=rope_rot_type,
        )
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        # MLP block
        self.norm2 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=0.0)
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rope_reg = self.rope_reg.pos_embed if self.rope_reg is not None else None
        x = x + self.layer_scale_1(self.attn(self.norm1(x), (self.rope.pos_embed, rope_reg)))
        x = x + self.layer_scale_2(self.mlp(self.norm2(x)))

        return x


# pylint: disable=invalid-name,too-many-instance-attributes
class RoPE_ViT5(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin, MaskedTokenRetentionMixin):
    block_group_regex = r"encoder\.block\.(\d+)"

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        image_size = self.size
        attention_dropout = 0.0
        dropout = 0.0
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", 1e-4)
        pre_norm: bool = self.config.get("pre_norm", False)
        post_norm: bool = self.config.get("post_norm", True)
        qkv_bias: bool = self.config.get("qkv_bias", False)
        qk_norm: bool = self.config.get("qk_norm", True)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        attn_pool_head: bool = self.config.get("attn_pool_head", False)
        attn_pool_num_heads: Optional[int] = self.config.get("attn_pool_num_heads", None)
        attn_pool_special_tokens: bool = self.config.get("attn_pool_special_tokens", False)
        norm_layer_type: str = self.config.get("norm_layer_type", "RMSNorm")
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-6)
        mlp_layer_type: str = self.config.get("mlp_layer_type", "FFN")
        act_layer_type: Optional[str] = self.config.get("act_layer_type", None)  # Default according to mlp type
        out_indices: Optional[list[int]] = self.config.get("out_indices", None)
        rope_rot_type: Literal["standard", "interleaved"] = self.config.get("rope_rot_type", "standard")
        rope_grid_indexing: Literal["ij", "xy"] = self.config.get("rope_grid_indexing", "ij")
        rope_grid_offset: int = self.config.get("rope_grid_offset", 0)
        rope_temperature: float = self.config.get("rope_temperature", 10000.0)
        rope_reg_temperature: float = self.config.get("rope_reg_temperature", 100.0)
        pt_grid_size: Optional[tuple[int, int]] = self.config.get("pt_grid_size", None)
        drop_path_rate: float = self.config["drop_path_rate"]

        if norm_layer_type == "LayerNorm":
            norm_layer = nn.LayerNorm
        elif norm_layer_type == "RMSNorm":
            norm_layer = nn.RMSNorm
        else:
            raise ValueError(f"Unknown norm_layer_type '{norm_layer_type}'")

        if mlp_layer_type == "FFN":
            mlp_layer = FFN
            act_layer = nn.GELU
        elif mlp_layer_type == "SwiGLU_FFN":
            mlp_layer = SwiGLU_FFN
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown mlp_layer_type '{mlp_layer_type}'")

        if act_layer_type is not None:
            act_layer = get_activation_module(act_layer_type)

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layer_scale_init_value = layer_scale_init_value
        self.num_reg_tokens = num_reg_tokens
        self.attn_pool_special_tokens = attn_pool_special_tokens
        self.norm_layer = norm_layer
        self.norm_layer_eps = norm_layer_eps
        self.mlp_layer = mlp_layer
        self.act_layer = act_layer
        self.out_indices = normalize_out_indices(out_indices, num_layers)
        self.rope_rot_type = rope_rot_type
        self.rope_grid_indexing = rope_grid_indexing
        self.rope_grid_offset = rope_grid_offset
        self.rope_temperature = rope_temperature
        self.rope_reg_temperature = rope_reg_temperature

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
            bias=not pre_norm,
        )
        self.patch_embed = PatchEmbed()

        seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.num_special_tokens = 0

        # Add a class token
        if class_token is True:
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.num_special_tokens += 1
        else:
            self.class_token = None

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
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
            grid_indexing=rope_grid_indexing,
            grid_offset=rope_grid_offset,
            pt_grid_size=self.pt_grid_size,
            rope_rot_type=rope_rot_type,
        )
        if self.num_reg_tokens == 0:
            self.rope_reg = None
        else:
            self.rope_reg = RoPE(
                hidden_dim // num_heads,
                temperature=self.rope_reg_temperature,
                grid_size=_register_rope_grid(self.num_reg_tokens),
                grid_indexing=rope_grid_indexing,
                grid_offset=0,
                rope_rot_type=rope_rot_type,
            )

        # Encoder
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            self.num_special_tokens,
            self.num_reg_tokens,
            dropout,
            attention_dropout,
            dpr,
            pre_norm=pre_norm,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            activation_layer=act_layer,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
            mlp_layer=mlp_layer,
            rope_rot_type=rope_rot_type,
        )

        if post_norm is True:
            self.norm = norm_layer(hidden_dim, eps=norm_layer_eps)
        else:
            self.norm = nn.Identity()

        if attn_pool_head is False:
            self.attn_pool = None
        else:
            if attn_pool_num_heads is None:
                attn_pool_num_heads = num_heads

            self.attn_pool = MultiHeadAttentionPool(hidden_dim, attn_pool_num_heads, mlp_dim, qkv_bias=True)

        num_return_stages = len(self.out_indices) if self.out_indices is not None else 1
        self.return_stages = [f"stage{stage_idx + 1}" for stage_idx in range(num_return_stages)]
        self.return_channels = [hidden_dim] * num_return_stages
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
            num_reg_tokens=self.num_reg_tokens,
            activation_layer=act_layer,
            grid_size=(image_size[0] // patch_size, image_size[1] // patch_size),
            rope_grid_indexing=rope_grid_indexing,
            rope_grid_offset=rope_grid_offset,
            rope_temperature=self.rope_temperature,
            rope_reg_temperature=self.rope_reg_temperature,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
            mlp_layer=mlp_layer,
            rope_rot_type=rope_rot_type,
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
            0,
            antialias=False,
        )

    def _get_rope_embed(self, H: int, W: int) -> RoPEPosEmbedType:
        if self.dynamic_size is False:
            rope = self.rope.pos_embed
        elif H == self.size[0] and W == self.size[1]:
            rope = self.rope.pos_embed
        else:
            rope = torch.concat(
                build_rotary_pos_embed(
                    self.hidden_dim // self.num_heads,
                    self.rope_temperature,
                    grid_size=(H // self.patch_size, W // self.patch_size),
                    grid_indexing=self.rope_grid_indexing,
                    grid_offset=self.rope_grid_offset,
                    pt_grid_size=self.pt_grid_size,
                ),
                dim=-1,
            ).to(self.rope.pos_embed.device, dtype=self.rope.pos_embed.dtype)

        if self.rope_reg is not None:
            rope_reg = self.rope_reg.pos_embed.to(rope.device, dtype=rope.dtype)
        else:
            rope_reg = None

        return (rope, rope_reg)

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad_(True)

        if unfreeze_features is True:
            if self.attn_pool is not None:
                for param in self.attn_pool.parameters():
                    param.requires_grad_(True)

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.encoder.set_causal_attention(is_causal)

    def transform_to_backbone(self) -> None:
        super().transform_to_backbone()
        self.norm = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        rope = self._get_rope_embed(H, W)
        if self.out_indices is None:
            xs = [self.encoder(x, rope)]
        else:
            xs = self.encoder.forward_features(x, rope, out_indices=self.out_indices)

        out: dict[str, torch.Tensor] = {}
        for stage_name, stage_x in zip(self.return_stages, xs, strict=True):
            stage_x = stage_x[:, self.num_special_tokens :]
            stage_x = stage_x.permute(0, 2, 1)
            B, C, _ = stage_x.size()
            stage_x = stage_x.reshape(B, C, H // self.patch_size, W // self.patch_size)
            out[stage_name] = stage_x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.conv_proj.parameters():
            param.requires_grad_(False)

        self.pos_embedding.requires_grad_(False)

        for idx, module in enumerate(self.encoder.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def masked_encoding_omission(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.Tensor] = None,
        return_all_features: bool = False,
        return_keys: Literal["all", "tokens", "embedding"] = "tokens",
    ) -> TokenOmissionResultType:
        H, W = x.shape[-2:]

        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Add pos embedding without special tokens
        pos_embedding = self._get_pos_embed(H, W)
        x = x + pos_embedding
        rope, rope_reg = self._get_rope_embed(H, W)

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

            rope_dim = rope.size(1)
            rope = rope.unsqueeze(0).repeat(x.size(0), 1, 1)
            rope = torch.gather(rope, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, rope_dim))

        rope_masked: RoPEPosEmbedType = (rope, rope_reg)

        # Append class and register tokens
        if self.class_token is not None:
            cls_token = self.class_token

            batch_class_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.concat((batch_class_token, x), dim=1)

        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens

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

        result: TokenOmissionResultType = {}
        if return_keys in ("all", "tokens"):
            result["tokens"] = x

        if return_keys in ("all", "embedding"):
            if return_all_features is True:
                x = x[..., -1]

            if self.attn_pool is not None:
                if self.attn_pool_special_tokens is False:
                    x = x[:, self.num_special_tokens :]

                x = self.attn_pool(x)
                result["embedding"] = x[:, 0]
            elif self.class_token is None:
                x = x[:, self.num_special_tokens :]
                result["embedding"] = x.mean(dim=1)
            else:
                result["embedding"] = x[:, self.num_reg_tokens]

        return result

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        H, W = x.shape[-2:]

        x = self.conv_proj(x)
        x = mask_tensor(x, mask, mask_token=mask_token, patch_factor=self.max_stride // self.stem_stride)

        # Reshape and permute the input tensor
        x = self.patch_embed(x)

        x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        x = self.encoder(x, self._get_rope_embed(H, W))
        x = self.norm(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            features = x[:, self.num_special_tokens :]
            features = features.permute(0, 2, 1)
            B, C, _ = features.size()
            features = features.reshape(B, C, H // self.patch_size, W // self.patch_size)
            result["features"] = features

        if return_keys in ("all", "embedding"):
            if self.attn_pool is not None:
                if self.attn_pool_special_tokens is False:
                    x = x[:, self.num_special_tokens :]

                x = self.attn_pool(x)
                result["embedding"] = x[:, 0]
            elif self.class_token is None:
                x = x[:, self.num_special_tokens :]
                result["embedding"] = x.mean(dim=1)
            else:
                result["embedding"] = x[:, self.num_reg_tokens]

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        x = self.encoder(x, self._get_rope_embed(H, W))
        x = self.norm(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        if self.attn_pool is not None:
            if self.attn_pool_special_tokens is False:
                x = x[:, self.num_special_tokens :]

            x = self.attn_pool(x)
            return x[:, 0]

        if self.class_token is None:
            x = x[:, self.num_special_tokens :]
            return x.mean(dim=1)

        # Classifier "token" as used by standard language architectures
        return x[:, self.num_reg_tokens]

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        old_size = self.size
        super().adjust_size(new_size)

        # Add back class tokens
        with torch.no_grad():
            pos_embedding = adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                0,
            )

        self.pos_embedding = nn.Parameter(pos_embedding)

        # Adjust RoPE
        self.rope = RoPE(
            self.hidden_dim // self.num_heads,
            temperature=self.rope_temperature,
            grid_size=(new_size[0] // self.patch_size, new_size[1] // self.patch_size),
            grid_indexing=self.rope_grid_indexing,
            grid_offset=self.rope_grid_offset,
            pt_grid_size=self.pt_grid_size,
            rope_rot_type=self.rope_rot_type,
            device=self.rope.pos_embed.device,
        )

        # Define adjusted decoder block
        self.decoder_block = partial(
            MAEDecoderBlock,
            16,
            num_special_tokens=self.num_special_tokens,
            num_reg_tokens=self.num_reg_tokens,
            activation_layer=self.act_layer,
            grid_size=(new_size[0] // self.patch_size, new_size[1] // self.patch_size),
            rope_grid_indexing=self.rope_grid_indexing,
            rope_grid_offset=self.rope_grid_offset,
            rope_temperature=self.rope_temperature,
            rope_reg_temperature=self.rope_reg_temperature,
            layer_scale_init_value=self.layer_scale_init_value,
            norm_layer=self.norm_layer,
            norm_layer_eps=self.norm_layer_eps,
            mlp_layer=self.mlp_layer,
            rope_rot_type=self.rope_rot_type,
        )


registry.register_model_config(
    "rope_vit5_reg4_t16",
    RoPE_ViT5,
    config={"patch_size": 16, **TINY, "num_reg_tokens": 4},
)
registry.register_model_config(
    "rope_vit5_reg4_s16",
    RoPE_ViT5,
    config={"patch_size": 16, **SMALL, "num_reg_tokens": 4, "drop_path_rate": 0.05},
)
registry.register_model_config(
    "rope_vit5_reg4_m16",
    RoPE_ViT5,
    config={"patch_size": 16, **MEDIUM, "num_reg_tokens": 4, "drop_path_rate": 0.1},
)
registry.register_model_config(
    "rope_vit5_reg4_b16",
    RoPE_ViT5,
    config={"patch_size": 16, **BASE, "num_reg_tokens": 4, "drop_path_rate": 0.2},
)
registry.register_model_config(
    "rope_vit5_reg4_so150m_p16",
    RoPE_ViT5,
    config={"patch_size": 16, **SO150, "num_reg_tokens": 4, "drop_path_rate": 0.25},
)
registry.register_model_config(
    "rope_vit5_reg4_l16",
    RoPE_ViT5,
    config={"patch_size": 16, **LARGE, "num_reg_tokens": 4, "drop_path_rate": 0.35},
)
registry.register_model_config(
    "rope_vit5_reg4_so400m_p16",
    RoPE_ViT5,
    config={"patch_size": 16, **SO400, "num_reg_tokens": 4, "drop_path_rate": 0.35},
)
registry.register_model_config(
    "rope_vit5_reg4_h16",
    RoPE_ViT5,
    config={"patch_size": 16, **HUGE, "num_reg_tokens": 4, "drop_path_rate": 0.4},
)
registry.register_model_config(  # From "Scaling Vision Transformers"
    "rope_vit5_reg4_g16",
    RoPE_ViT5,
    config={"patch_size": 16, **GIANT, "num_reg_tokens": 4, "drop_path_rate": 0.4},
)
