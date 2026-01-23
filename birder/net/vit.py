"""
ViT, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

Paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
https://arxiv.org/abs/2010.11929
and
Paper "Vision Transformers Need Registers", https://arxiv.org/abs/2309.16588
and
Paper "Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design", https://arxiv.org/abs/2305.13035
"""

# Reference license: BSD 3-Clause and Apache-2.0

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
from birder.net._vit_configs import register_vit_configs
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import TokenRetentionResultType
from birder.net.base import normalize_out_indices


def adjust_position_embedding(
    pos_embedding: torch.Tensor,
    old_base_size: tuple[int, int],
    new_base_size: tuple[int, int],
    num_prefix_tokens: int,
    antialias: bool = True,
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py
    """

    pos_embedding_prefix = pos_embedding[:, :num_prefix_tokens]
    pos_embedding = pos_embedding[:, num_prefix_tokens:]

    # Interpolation
    embed_dim = pos_embedding.shape[-1]
    orig_dtype = pos_embedding.dtype
    pos_embedding = pos_embedding.float()  # Interpolate needs float32
    pos_embedding = pos_embedding.reshape(1, old_base_size[0], old_base_size[1], -1).permute(0, 3, 1, 2)
    pos_embedding = F.interpolate(pos_embedding, size=new_base_size, mode="bicubic", antialias=antialias)
    pos_embedding = pos_embedding.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    pos_embedding = pos_embedding.to(orig_dtype)

    # Add back special tokens
    return torch.concat([pos_embedding_prefix, pos_embedding], dim=1)


class PatchEmbed(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is equivalent (in output) to: x.flatten(2).transpose(1, 2)
        """

        n, hidden_dim, h, w = x.size()
        x = x.reshape(n, hidden_dim, h * w)

        # (n, hidden_dim, (h * w)) -> (n, (h * w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

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

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head self-attention to the input sequence

        This module implements scaled dot-product attention over x and returns the
        projected output.

        Parameters
        ----------
        x
            Input tensor of shape (B, N, C) where B is batch size, N is sequence length,
            and C is embedding dimension.
        need_weights
            If True, also return attention weights computed explicitly. If False, uses
            torch.nn.functional.scaled_dot_product_attention and returns None for attention weights.
        average_attn_weights
            If True and need_weights is True, average attention weights across heads
            to shape (B, N, N). If False, return per-head weights of shape (B, num_heads, N, N).
        is_causal
            If True, apply a causal (upper-triangular) mask so positions cannot attend to future tokens.

        Returns
        -------
        A tuple containing two elements:
            - output: Tensor of shape (B, N, C)
            - attn_weights: If need_weights is True attention weights, otherwise, None.
        """

        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_weights: Optional[torch.Tensor] = None
        if need_weights is True:
            # Compute attention manually to get weights
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if is_causal is True:
                causal_mask = torch.triu(
                    torch.full((N, N), float("-inf"), dtype=attn.dtype, device=attn.device),
                    diagonal=1,
                )
                attn = attn + causal_mask

            attn = attn.softmax(dim=-1)
            attn_weights = attn
            attn = self.attn_drop(attn)
            x = attn @ v

            if average_attn_weights is True:
                # Average across heads: (B, num_heads, N, N) -> (B, N, N)
                attn_weights = attn_weights.mean(dim=1)
        else:
            x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=is_causal, scale=self.scale
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return (x, attn_weights)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: Optional[int],
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
    ) -> None:
        super().__init__()
        self.need_attn = False
        self.is_causal = False

        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Attention block
        self.norm1 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.attn = Attention(
            hidden_dim,
            num_heads=num_heads,
            attn_drop=attention_dropout,
            proj_drop=0.0,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
        )

        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        # MLP block
        self.norm2 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=dropout)
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.size()}")
        attn_out, _ = self.attn(
            self.norm1(x),
            need_weights=self.need_attn,
            average_attn_weights=False,
            is_causal=self.is_causal,
        )
        x = x + self.drop_path(self.layer_scale_1(attn_out))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x

    def set_need_attn(self, need_attn: bool = True) -> None:
        self.need_attn = need_attn

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.is_causal = is_causal


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
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
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_block(x)
        return self.block(x)

    def forward_features(self, x: torch.Tensor, out_indices: Optional[list[int]] = None) -> list[torch.Tensor]:
        x = self.pre_block(x)

        out_indices_set = set(out_indices) if out_indices is not None else None
        xs = []
        for idx, blk in enumerate(self.block):
            x = blk(x)
            if out_indices_set is None or idx in out_indices_set:
                xs.append(x)

        return xs

    def set_need_attn(self, need_attn: bool = True) -> None:
        for b in self.block:
            b.set_need_attn(need_attn)

    def set_causal_attention(self, is_causal: bool = True) -> None:
        for b in self.block:
            b.set_causal_attention(is_causal)


# pylint: disable=too-many-instance-attributes
class ViT(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin, MaskedTokenRetentionMixin):
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
        pos_embed_special_tokens: bool = self.config.get("pos_embed_special_tokens", True)
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        pre_norm: bool = self.config.get("pre_norm", False)
        post_norm: bool = self.config.get("post_norm", True)
        qkv_bias: bool = self.config.get("qkv_bias", True)
        qk_norm: bool = self.config.get("qk_norm", False)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        attn_pool_head: bool = self.config.get("attn_pool_head", False)
        attn_pool_num_heads: Optional[int] = self.config.get("attn_pool_num_heads", None)
        attn_pool_special_tokens: bool = self.config.get("attn_pool_special_tokens", False)
        norm_layer_type: str = self.config.get("norm_layer_type", "LayerNorm")
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-6)
        mlp_layer_type: str = self.config.get("mlp_layer_type", "FFN")
        act_layer_type: Optional[str] = self.config.get("act_layer_type", None)  # Default according to mlp type
        out_indices: Optional[list[int]] = self.config.get("out_indices", None)
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
        self.pos_embed_special_tokens = pos_embed_special_tokens
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_reg_tokens = num_reg_tokens
        self.attn_pool_special_tokens = attn_pool_special_tokens
        self.out_indices = normalize_out_indices(out_indices, num_layers)
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
            if pos_embed_special_tokens is True:
                seq_length += 1
        else:
            self.class_token = None

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
            self.num_special_tokens += self.num_reg_tokens
            if pos_embed_special_tokens is True:
                seq_length += self.num_reg_tokens
        else:
            self.reg_tokens = None

        # Add positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
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
            EncoderBlock,
            16,
            mlp_dim=None,
            dropout=0,
            attention_dropout=0,
            drop_path=0,
            activation_layer=act_layer,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
            mlp_layer=mlp_layer,
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
            self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            antialias=False,
        )

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

        if self.pos_embed_special_tokens is False:
            x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + self._get_pos_embed(H, W)

        if self.out_indices is None:
            xs = [self.encoder(x)]
        else:
            xs = self.encoder.forward_features(x, out_indices=self.out_indices)

        out: dict[str, torch.Tensor] = {}
        for stage_name, stage_x in zip(self.return_stages, xs):
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

    # pylint: disable=too-many-branches
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
        if self.pos_embed_special_tokens is True:
            x = x + pos_embedding[:, self.num_special_tokens :, :]
        else:
            x = x + pos_embedding

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

        # Append class and register tokens
        if self.class_token is not None:
            if self.pos_embed_special_tokens is True:
                cls_token = self.class_token + pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
            else:
                cls_token = self.class_token

            batch_class_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.concat((batch_class_token, x), dim=1)

        if self.reg_tokens is not None:
            if self.pos_embed_special_tokens is True:
                reg_tokens = self.reg_tokens + pos_embedding[:, 0 : self.num_reg_tokens, :]
            else:
                reg_tokens = self.reg_tokens

            batch_reg_tokens = reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        # Apply transformer
        if return_all_features is True:
            xs = self.encoder.forward_features(x)
            xs[-1] = self.norm(xs[-1])
            x = torch.stack(xs, dim=-1)
        else:
            x = self.encoder(x)
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

        if self.pos_embed_special_tokens is False:
            x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + self._get_pos_embed(H, W)

        x = self.encoder(x)
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

        if self.pos_embed_special_tokens is False:
            x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + self._get_pos_embed(H, W)

        x = self.encoder(x)
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

        if self.pos_embed_special_tokens is True:
            num_prefix_tokens = self.num_special_tokens
        else:
            num_prefix_tokens = 0

        # Add back class tokens
        with torch.no_grad():
            pos_embedding = adjust_position_embedding(
                # On rounding error see: https://github.com/facebookresearch/dino/issues/8
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                num_prefix_tokens,
            )

        self.pos_embedding = nn.Parameter(pos_embedding)


# Register model configs (side effects)
register_vit_configs(ViT)

registry.register_weights(
    "vit_l16_mim_200",
    {
        "url": "https://huggingface.co/birder-project/vit_l16_mim/resolve/main",
        "description": (
            "ViT l16 image encoder pre-trained using Masked Image Modeling (MIM) for 200 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1157.1,
                "sha256": "7fc5b342347d8349aaf5f069a47efd441b646f8542821ed2e30b47a7da72917a",
            },
        },
        "net": {"network": "vit_l16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_l16_mim_400",
    {
        "url": "https://huggingface.co/birder-project/vit_l16_mim/resolve/main",
        "description": (
            "ViT l16 image encoder pre-trained using Masked Image Modeling (MIM) for 400 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1157.1,
                "sha256": "9b5c4e2538ea40edd60d8831d3807b543290dc2db44d537e60e44a341b47e54e",
            },
        },
        "net": {"network": "vit_l16", "tag": "mim"},
    },
)
registry.register_weights(  # BioCLIP v2: https://arxiv.org/abs/2505.23883
    "vit_l14_pn_bioclip-v2",
    {
        "url": "https://huggingface.co/birder-project/vit_l14_pn_bioclip-v2/resolve/main",
        "description": (
            "ViT l14 image encoder pre-trained by Imageomics using CLIP on the TreeOfLife-200M dataset. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1156.6,
                "sha256": "6cd7bd6993762590891fe2b41db1649cde5a0c4de5a7f341672f8856ed529d07",
            },
        },
        "net": {"network": "vit_l14_pn", "tag": "bioclip-v2"},
    },
)
registry.register_weights(  # OpenAI CLIP: https://arxiv.org/abs/2103.00020
    "vit_l14_pn_quick_gelu_openai-clip",
    {
        "url": "https://huggingface.co/birder-project/vit_l14_pn_quick_gelu_openai-clip/resolve/main",
        "description": (
            "ViT l14 image encoder pre-trained by OpenAI using CLIP. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1159.7,
                "sha256": "2c7462390956d8942de0df21d9d1a43cf53fdbe3a3570a1add64d859313a0bee",
            },
        },
        "net": {"network": "vit_l14_pn_quick_gelu", "tag": "openai-clip"},
    },
)
registry.register_weights(  # SigLIP 2: https://arxiv.org/abs/2502.14786
    "vit_so400m_p14_ap_siglip-v2-webli",
    {
        "url": "https://huggingface.co/birder-project/vit_so400m_p14_ap_siglip-v2-webli/resolve/main",
        "description": (
            "ViT SO400m image encoder pre-trained by Google using SigLIP. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1631.6,
                "sha256": "f8ac3bdf028d17a2ee2673f58b51cffa5c696edef44c92092299d970607c7be6",
            },
        },
        "net": {"network": "vit_so400m_p14_ap", "tag": "siglip-v2-webli"},
    },
)

# With registers
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA"
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 146.2,
                "sha256": "0f5cd4e0acb44d1e429bbed342c60bf22087ecd1d7112363c3ceb909dcd9d547",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa"},
    },
)
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa-inat21-256px",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa-inat21/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA"
            "then fine-tuned on the iNaturalist 2021 dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 166.8,
                "sha256": "e9b83e90c284877c859e92a05a35ff25884a06d3fd006d90ee576d58f71d3251",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa-inat21-256px"},
    },
)
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa-inat21",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa-inat21/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA"
            "then fine-tuned on the iNaturalist 2021 dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 167.4,
                "sha256": "7fde7375f5f9165114561f6288cdf086ba8b6635251304de08bd01883bb7a2da",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa-inat21"},
    },
)
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa-imagenet21k",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa-imagenet21k/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA then fine-tuned on the ImageNet-21K dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 184.2,
                "sha256": "da47dc6bd4f41c347235beba92657b66148180141d0bd629169e84449b629fbb",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa-imagenet21k"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim_200",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim/resolve/main",
        "description": (
            "ViT reg4 b16 image encoder pre-trained using Masked Image Modeling (MIM) for 200 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 327.4,
                "sha256": "c7ec433c01e1dc0d6100cafc29fa88155a0d65f4b42afa9cc252b77485a566a7",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim_300",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim/resolve/main",
        "description": (
            "ViT reg4 b16 image encoder pre-trained using Masked Image Modeling (MIM) for 300 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 327.4,
                "sha256": "b0e5e2b24ea7a8d2be246df43c9d8092354f6ee81e88c6cdd7c52d8e38ed44a4",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim-intermediate-il-common",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim-intermediate-il-common/resolve/main",
        "description": (
            "ViT reg4 b16 model with MIM pretraining and intermediate training, "
            "then fine-tuned on the il-common dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 328.7,
                "sha256": "3a15b95285cd4435b601ef058839f422cdce8f68cca50de9353e1ac2bcb65f9a",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim-intermediate-il-common"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim-intermediate-arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim-intermediate-arabian-peninsula/resolve/main",
        "description": (
            "ViT reg4 b16 model with MIM pretraining and intermediate training, "
            "then fine-tuned on the arabian-peninsula dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 330.7,
                "sha256": "78dbf578ebe7d5761705231e16fef280b14905a94f18879167c96df3e59d13a5",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim-intermediate-arabian-peninsula"},
    },
)
registry.register_weights(  # DINO v2: https://arxiv.org/abs/2304.07193
    "vit_reg4_l14_nps_ls_dino-v2-lvd142m",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_l14_nps_ls_dino-v2-lvd142m/resolve/main",
        "description": (
            "ViT reg4 l14 image encoder pre-trained by Facebook AI using DINOv2. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (518, 518),
        "formats": {
            "pt": {
                "file_size": 1161.2,
                "sha256": "441721029ca0ef85582bc8822ec91d780ee442eb3d06b04fb5e4662c9317b52d",
            },
        },
        "net": {"network": "vit_reg4_l14_nps_ls", "tag": "dino-v2-lvd142m"},
    },
)
