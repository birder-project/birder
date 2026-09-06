"""
Paper "Scaling Vision with Sparse Mixture of Experts", https://arxiv.org/abs/2106.05974
and
Paper "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
https://arxiv.org/abs/2401.06066
and
Paper "DeepSeek-V3 Technical Report", https://arxiv.org/abs/2412.19437
and
Paper "Mixture-of-Experts with Expert Choice Routing", https://arxiv.org/abs/2202.09368
"""

# Reference license: Apache-2.0

import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import FFN
from birder.layers import BaseSparseMoE_FFN
from birder.layers import EfficientProbing
from birder.layers import LayerNorm2d
from birder.layers import LayerScale
from birder.layers import MoE_FFN
from birder.layers import MoESpec
from birder.layers import MultiHeadAttentionPool
from birder.layers import SwiGLU_FFN
from birder.layers import VMoE_FFN
from birder.layers.activations import get_activation_module
from birder.layers.moe import MoETrainingOutputType
from birder.layers.moe import _empty_moe_training_output
from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import LARGE
from birder.net._vit_configs import MEDIUM
from birder.net._vit_configs import SMALL
from birder.net._vit_configs import SO150
from birder.net._vit_configs import TINY
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import TokenRetentionResultType
from birder.net.base import stochastic_depth_rates
from birder.net.vit import Attention
from birder.net.vit import EncoderBlock as ViTEncoderBlock
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding
from birder.net.vit import hMLPStem

logger = logging.getLogger(__name__)

# Keep this payload flat and identical across MoE implementations so both checkpoint variants can return tensors only.
# Loss-based routers leave expert_loads empty, auxiliary-loss-free routers leave the scalar loss fields at zero.
_MOE_LOSS_KEYS = ("auxiliary_loss", "g_shard_loss", "importance_loss", "load_loss")
_MOE_TRAINING_OUTPUT_KEYS = (*_MOE_LOSS_KEYS, "expert_loads")

V_MOE_SMALL = {"num_layers": 8, "num_heads": 8, "hidden_dim": 512, "mlp_dim": 2048, "drop_path_rate": 0.0}


def _resolve_moe_layers(
    num_layers: int,
    *,
    moe_layers: Optional[list[int]],
    moe_every_n_layers: Optional[int],
    moe_last_n_layers: Optional[int],
    moe_last_n_layers_stride: int = 1,
) -> list[int]:
    assert (
        sum(x is not None for x in (moe_layers, moe_every_n_layers, moe_last_n_layers)) == 1
    ), "Exactly one of moe_layers, moe_every_n_layers, or moe_last_n_layers must be set"
    assert (
        moe_last_n_layers is not None or moe_last_n_layers_stride == 1
    ), "moe_last_n_layers_stride is only supported with moe_last_n_layers"

    if moe_layers is not None:
        assert all(
            idx < num_layers for idx in moe_layers
        ), f"moe_layers indices are zero-based and must be less than num_layers={num_layers}"
        return moe_layers

    if moe_every_n_layers is not None:
        return list(range(moe_every_n_layers - 1, num_layers, moe_every_n_layers))

    assert moe_last_n_layers is not None
    assert (
        moe_last_n_layers - 1
    ) * moe_last_n_layers_stride < num_layers, "moe_last_n_layers select more layers than are available"

    last_layers = range(num_layers - 1, -1, -moe_last_n_layers_stride)
    return list(reversed(list(last_layers)[:moe_last_n_layers]))


def _aggregate_moe_training_outputs(
    training_outputs: list[MoETrainingOutputType], ref: torch.Tensor
) -> MoETrainingOutputType:
    if len(training_outputs) == 0:
        return _empty_moe_training_output(ref)

    expert_loads = [output["expert_loads"] for output in training_outputs if output["expert_loads"].size(0) > 0]
    return {
        "auxiliary_loss": torch.stack([output["auxiliary_loss"] for output in training_outputs]).sum(),
        "g_shard_loss": torch.stack([output["g_shard_loss"] for output in training_outputs]).sum(),
        "importance_loss": torch.stack([output["importance_loss"] for output in training_outputs]).sum(),
        "load_loss": torch.stack([output["load_loss"] for output in training_outputs]).sum(),
        "expert_loads": (
            torch.concat(expert_loads) if len(expert_loads) > 0 else ref.new_empty((0, 0), dtype=torch.int64)
        ),
    }


def _unpack_moe_training_output(flat_output: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> MoETrainingOutputType:
    auxiliary_loss, g_shard_loss, importance_loss, load_loss, expert_loads = flat_output
    return {
        "auxiliary_loss": auxiliary_loss,
        "g_shard_loss": g_shard_loss,
        "importance_loss": importance_loss,
        "load_loss": load_loss,
        "expert_loads": expert_loads,
    }


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        projection_dropout: float,
        drop_path: float,
        num_special_tokens: int = 0,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_norm: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.attn = Attention(
            hidden_dim,
            num_heads=num_heads,
            attn_drop=attention_dropout,
            proj_drop=projection_dropout,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_norm=attn_norm,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
        )

        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        self.norm2 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, dropout=dropout)
        self.is_sparse_moe = isinstance(self.mlp, BaseSparseMoE_FFN)
        self.has_special_token_experts = isinstance(self.mlp, MoE_FFN) and self.mlp.has_special_token_experts
        self.num_special_tokens = num_special_tokens
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

        # Weight initialization
        if self.attn.qkv.bias is not None:
            nn.init.zeros_(self.attn.qkv.bias)

        nn.init.xavier_uniform_(self.attn.proj.weight)
        if self.attn.proj.bias is not None:
            nn.init.zeros_(self.attn.proj.bias)

        if isinstance(self.mlp, FFN):
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, std=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        token_mask: Optional[torch.Tensor] = None,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        attn_out, _ = self.attn(self.norm1(x), is_causal=is_causal, attn_mask=attn_mask)
        x = x + self.drop_path(self.layer_scale_1(attn_out))
        y = self.norm2(x)
        if self.is_sparse_moe is True:
            moe_kwargs: dict[str, Any] = {"token_mask": token_mask}
            if self.has_special_token_experts is True:
                moe_kwargs["num_special_tokens"] = self.num_special_tokens

            if return_moe_training_output is True:
                y, moe_training_output = self.mlp(y, return_moe_training_output=True, **moe_kwargs)
                x = x + self.drop_path(self.layer_scale_2(y))
                return (x, moe_training_output)

            y = self.mlp(y, **moe_kwargs)
        else:
            y = self.mlp(y)

        x = x + self.drop_path(self.layer_scale_2(y))
        if return_moe_training_output is True:
            return (x, _empty_moe_training_output(x))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        moe_layers: list[int],
        dropout: float,
        attention_dropout: float,
        projection_dropout: float,
        dpr: list[float],
        num_special_tokens: int = 0,
        pre_norm: bool = False,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_norm: bool = False,
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
        moe_ffn_type: Literal["VMoE_FFN", "MoE_FFN"] = "VMoE_FFN",
        moe_expert_width: Optional[int] = None,
        moe_ffn_bias: bool = False,
        moe_dropout: float = 0.0,
        moe_num_experts: int = 8,
        moe_num_shared_experts: int = 1,
        moe_num_special_token_experts: int = 0,
        moe_routed_scaling_factor: float = 1.0,
        moe_routing_type: Literal["token_choice", "expert_choice"] = "token_choice",
        moe_top_k: int = 2,
        router_bias_update_speed: float = 0.001,
        moe_expert_choice_capacity_factor: float = 2.0,
        moe_capacity_factor: float = 1.05,
        moe_eval_capacity_factor: float = 4.0,
        moe_capacity_multiple_of: Optional[int] = 4,
        router_noise_std: float = 1.0,
        router_g_shard_loss_weight: float = 0.0,
        router_importance_loss_weight: float = 0.005,
        router_load_loss_weight: float = 0.005,
    ) -> None:
        super().__init__()
        self.is_causal = False
        self.grad_checkpointing = False
        self.grad_checkpointing_segments: Optional[int] = None
        self.grad_checkpointing_preserve_rng_state = True
        self.grad_checkpointing_use_reentrant = False
        self.has_expert_choice_routing = moe_ffn_type == "MoE_FFN" and moe_routing_type == "expert_choice"

        if moe_ffn_type == "VMoE_FFN":
            self.moe_spec = MoESpec(has_auxiliary_loss=True, requires_expert_bias_update=False)
            if moe_expert_width is not None:
                raise ValueError("moe_expert_width is only supported with moe_ffn_type='MoE_FFN'")

            moe_mlp_dim = mlp_dim
            moe_mlp_layer: Callable[..., nn.Module] = partial(
                VMoE_FFN,
                act_layer=activation_layer,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                capacity_factor=moe_capacity_factor,
                eval_capacity_factor=moe_eval_capacity_factor,
                capacity_multiple_of=moe_capacity_multiple_of,
                router_noise_std=router_noise_std,
                router_g_shard_loss_weight=router_g_shard_loss_weight,
                router_importance_loss_weight=router_importance_loss_weight,
                router_load_loss_weight=router_load_loss_weight,
            )
        elif moe_ffn_type == "MoE_FFN":
            self.moe_spec = MoESpec(
                has_auxiliary_loss=False,
                requires_expert_bias_update=moe_routing_type == "token_choice",
            )
            if moe_expert_width is None:
                moe_expert_width = mlp_dim

            moe_mlp_dim = moe_expert_width
            moe_mlp_layer = partial(
                MoE_FFN,
                bias=moe_ffn_bias,
                num_routed_experts=moe_num_experts - moe_num_shared_experts - moe_num_special_token_experts,
                num_shared_experts=moe_num_shared_experts,
                num_special_token_experts=moe_num_special_token_experts,
                routed_scaling_factor=moe_routed_scaling_factor,
                routing_type=moe_routing_type,
                top_k=moe_top_k,
                router_bias_update_speed=router_bias_update_speed,
                expert_choice_capacity_factor=moe_expert_choice_capacity_factor,
            )
        else:
            raise ValueError(f"Unknown moe_ffn_type '{moe_ffn_type}'")

        dense_mlp_layer: Callable[..., nn.Module] = partial(mlp_layer, act_layer=activation_layer)
        pre_layers = []
        if dropout > 0.0:
            pre_layers.append(nn.Dropout(dropout))
        if pre_norm is True:
            pre_layers.append(norm_layer(hidden_dim, eps=norm_layer_eps))

        self.pre_block = nn.Sequential(*pre_layers)
        layers = []
        for i in range(num_layers):
            if i in moe_layers:
                block_mlp_dim = moe_mlp_dim
                mlp_dropout = moe_dropout
                mlp_layer = moe_mlp_layer
            else:
                block_mlp_dim = mlp_dim
                mlp_dropout = dropout
                mlp_layer = dense_mlp_layer

            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    block_mlp_dim,
                    mlp_dropout,
                    attention_dropout,
                    projection_dropout,
                    dpr[i],
                    num_special_tokens=num_special_tokens,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm_layer,
                    norm_layer_eps=norm_layer_eps,
                    mlp_layer=mlp_layer,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_norm=attn_norm,
                )
            )

        self.block = nn.ModuleList(layers)
        self._moe_ffns = tuple(blk.mlp for blk in self.block if isinstance(blk.mlp, MoE_FFN))

    def _prepare_attention_mask(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], bool]:
        if attn_mask is None:
            return (None, self.is_causal)

        if attn_mask.dtype == torch.bool:
            if self.is_causal is True:
                seq_len = x.size(1)
                causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=attn_mask.device).triu_(
                    diagonal=1
                )
                mask_shape = torch.broadcast_shapes(attn_mask.shape, causal_mask.shape)
                additive_mask = torch.zeros(mask_shape, dtype=x.dtype, device=attn_mask.device)
                additive_mask.masked_fill_(causal_mask, float("-inf"))
                additive_mask.masked_fill_(~attn_mask, float("-inf"))
            else:
                additive_mask = torch.full_like(attn_mask, float("-inf"), dtype=x.dtype)
                additive_mask.masked_fill_(attn_mask, 0.0)

            attn_mask = additive_mask
        elif self.is_causal is True:
            seq_len = x.size(1)
            causal_bias = torch.full(
                (seq_len, seq_len), float("-inf"), dtype=attn_mask.dtype, device=attn_mask.device
            ).triu_(diagonal=1)
            attn_mask = attn_mask + causal_bias

        return (attn_mask, False)

    def _checkpoint_block_range(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        training_outputs: list[MoETrainingOutputType] = []
        for blk in self.block[start:end]:
            if blk.is_sparse_moe is True:
                x, moe_training_output = blk(
                    x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask, return_moe_training_output=True
                )
                training_outputs.append(moe_training_output)
            else:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask)

        moe_training_output = _aggregate_moe_training_outputs(training_outputs, x)
        return (x, *(moe_training_output[key] for key in _MOE_TRAINING_OUTPUT_KEYS))

    def _checkpoint_blocks_with_moe(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, MoETrainingOutputType]:
        if self.grad_checkpointing_segments is None:
            segments = len(self.block)
        else:
            segments = min(self.grad_checkpointing_segments, len(self.block))

        segment_size = len(self.block) // segments
        training_outputs: list[MoETrainingOutputType] = []
        start = 0
        for segment_idx in range(segments):
            end = start + segment_size if segment_idx < segments - 1 else len(self.block)
            block_range = partial(
                self._checkpoint_block_range,
                start=start,
                end=end,
                attn_mask=attn_mask,
                is_causal=is_causal,
                token_mask=token_mask,
            )
            if segment_idx < segments - 1:
                checkpoint_output = checkpoint(
                    block_range,
                    x,
                    use_reentrant=self.grad_checkpointing_use_reentrant,
                    preserve_rng_state=self.grad_checkpointing_preserve_rng_state,
                )
            else:
                checkpoint_output = block_range(x)

            x = checkpoint_output[0]
            training_outputs.append(_unpack_moe_training_output(checkpoint_output[1:]))
            start = end

        return (x, _aggregate_moe_training_outputs(training_outputs, x))

    def _checkpoint_blocks(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.grad_checkpointing_segments is None:
            segments = len(self.block)
        else:
            segments = min(self.grad_checkpointing_segments, len(self.block))

        if attn_mask is not None or is_causal is True or token_mask is not None:
            blocks = tuple(
                partial(block, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask) for block in self.block
            )
        else:
            blocks = self.block

        return checkpoint_sequential(
            blocks,
            segments,
            x,
            use_reentrant=self.grad_checkpointing_use_reentrant,
            preserve_rng_state=self.grad_checkpointing_preserve_rng_state,
        )

    def update_moe_expert_biases(self, expert_loads: torch.Tensor) -> None:
        for moe_ffn, expert_load in zip(self._moe_ffns, expert_loads, strict=True):
            moe_ffn.update_expert_bias(expert_load)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        x = self.pre_block(x)
        attn_mask, is_causal = self._prepare_attention_mask(x, attn_mask)
        if self.grad_checkpointing is True and torch.is_grad_enabled() is True and not torch.jit.is_scripting():
            if return_moe_training_output is True:
                return self._checkpoint_blocks_with_moe(
                    x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask
                )

            return self._checkpoint_blocks(x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask)

        training_outputs: list[MoETrainingOutputType] = []
        for blk in self.block:
            if return_moe_training_output is True and blk.is_sparse_moe is True:
                x, moe_training_output = blk(
                    x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask, return_moe_training_output=True
                )
                training_outputs.append(moe_training_output)
            else:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask)

        if return_moe_training_output is True:
            return (x, _aggregate_moe_training_outputs(training_outputs, x))

        return x

    def forward_features(
        self,
        x: torch.Tensor,
        out_indices: Optional[list[int]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_moe_training_output: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], MoETrainingOutputType]:
        x = self.pre_block(x)
        attn_mask, is_causal = self._prepare_attention_mask(x, attn_mask)

        out_indices_set = set(out_indices) if out_indices is not None else None
        xs = []
        training_outputs: list[MoETrainingOutputType] = []
        for idx, blk in enumerate(self.block):
            if return_moe_training_output is True and blk.is_sparse_moe is True:
                x, moe_training_output = blk(
                    x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask, return_moe_training_output=True
                )
                training_outputs.append(moe_training_output)
            else:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal, token_mask=token_mask)

            if out_indices_set is None or idx in out_indices_set:
                xs.append(x)

        if return_moe_training_output is True:
            return (xs, _aggregate_moe_training_outputs(training_outputs, x))

        return xs

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        if enable is True:
            logger.debug(
                f"Enabling gradient checkpointing: segments={segments}, "
                f"preserve_rng_state={preserve_rng_state}, use_reentrant={use_reentrant}"
            )
        else:
            logger.debug("Disabling gradient checkpointing")

        self.grad_checkpointing = enable
        self.grad_checkpointing_segments = segments
        self.grad_checkpointing_preserve_rng_state = preserve_rng_state
        self.grad_checkpointing_use_reentrant = use_reentrant

    def set_causal_attention(self, is_causal: bool = True) -> None:
        if is_causal is True and self.has_expert_choice_routing is True:
            raise ValueError("Expert-choice routing does not support causal attention")

        self.is_causal = is_causal


class ViT_MoE(PreTrainEncoder, MaskedTokenOmissionMixin, MaskedTokenRetentionMixin):
    scriptable = False
    block_group_regex = r"encoder\.block\.(\d+)"

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
        abs_pos_embed: bool = self.config.get("abs_pos_embed", True)
        pos_embed_special_tokens: bool = self.config.get("pos_embed_special_tokens", True)
        pos_embed_interpolation_mode: Literal["bilinear", "bicubic"] = self.config.get(
            "pos_embed_interpolation_mode", "bicubic"
        )
        pos_embed_antialias: bool = self.config.get("pos_embed_antialias", False)  # Controls forward pass only
        patch_size: int = self.config["patch_size"]
        stem_type: Literal["patchify", "hmlp"] = self.config.get("stem_type", "patchify")
        stem_norm_layer_type: Optional[Literal["BatchNorm2d", "LayerNorm2d"]] = self.config.get(
            "stem_norm_layer_type", None
        )
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        pre_norm: bool = self.config.get("pre_norm", False)
        post_norm: bool = self.config.get("post_norm", True)
        norm_after_pool: bool = self.config.get("norm_after_pool", False)
        qkv_bias: bool = self.config.get("qkv_bias", True)
        qk_norm: bool = self.config.get("qk_norm", False)
        attn_norm: bool = self.config.get("attn_norm", False)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        attn_pool_head: bool = self.config.get("attn_pool_head", False)
        attn_pool_type: str = self.config.get("attn_pool_type", "MultiHeadAttentionPool")
        attn_pool_num_heads: Optional[int] = self.config.get("attn_pool_num_heads", None)
        attn_pool_special_tokens: bool = self.config.get("attn_pool_special_tokens", False)
        attn_pool_norm_eps: float = self.config.get("attn_pool_norm_eps", 1e-5)
        attn_pool_act_layer_type: str = self.config.get("attn_pool_act_layer_type", "gelu")
        norm_layer_type: str = self.config.get("norm_layer_type", "LayerNorm")
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-6)
        mlp_layer_type: str = self.config.get("mlp_layer_type", "FFN")
        mlp_head = self.config.get("mlp_head", False)
        act_layer_type: Optional[str] = self.config.get("act_layer_type", None)  # Default according to mlp type
        dropout: float = self.config.get("dropout", 0.0)
        attention_dropout: float = self.config.get("attention_dropout", 0.0)
        projection_dropout: float = self.config.get("projection_dropout", 0.0)
        drop_path_rate: float = self.config["drop_path_rate"]

        moe_layers = _resolve_moe_layers(
            num_layers,
            moe_layers=self.config.get("moe_layers", None),
            moe_every_n_layers=self.config.get("moe_every_n_layers", None),
            moe_last_n_layers=self.config.get("moe_last_n_layers", None),
            moe_last_n_layers_stride=self.config.get("moe_last_n_layers_stride", 1),
        )
        moe_ffn_type: Literal["VMoE_FFN", "MoE_FFN"] = self.config.get("moe_ffn_type", "VMoE_FFN")
        moe_expert_width: Optional[int] = self.config.get("moe_expert_width", None)
        moe_ffn_bias: bool = self.config.get("moe_ffn_bias", False)
        moe_dropout: float = self.config.get("moe_dropout", 0.0)
        moe_num_experts: int = self.config.get("moe_num_experts", 8)
        moe_num_shared_experts: int = self.config.get("moe_num_shared_experts", 0)
        moe_num_special_token_experts: int = self.config.get("moe_num_special_token_experts", 0)
        moe_routed_scaling_factor: float = self.config.get("moe_routed_scaling_factor", 1.0)
        moe_routing_type: Literal["token_choice", "expert_choice"] = self.config.get("moe_routing_type", "token_choice")
        moe_top_k: int = self.config.get("moe_top_k", 2)
        router_bias_update_speed: float = self.config.get("router_bias_update_speed", 0.001)
        moe_expert_choice_capacity_factor: float = self.config.get("moe_expert_choice_capacity_factor", 2.0)
        moe_capacity_factor: float = self.config.get("moe_capacity_factor", 1.05)
        moe_eval_capacity_factor: float = self.config.get("moe_eval_capacity_factor", 4.0)
        moe_capacity_multiple_of: Optional[int] = self.config.get("moe_capacity_multiple_of", 4)
        router_noise_std: float = self.config.get("router_noise_std", 1.0)
        router_g_shard_loss_weight: float = self.config.get("router_g_shard_loss_weight", 0.0)
        router_importance_loss_weight: float = self.config.get("router_importance_loss_weight", 0.005)
        router_load_loss_weight: float = self.config.get("router_load_loss_weight", 0.005)

        if pos_embed_interpolation_mode not in ("bilinear", "bicubic"):
            raise ValueError(f"Unknown pos_embed_interpolation_mode '{pos_embed_interpolation_mode}'")

        if stem_type == "patchify":
            if stem_norm_layer_type is not None:
                raise ValueError("stem_norm_layer_type is only supported with stem_type='hmlp'")

            self.conv_proj = nn.Conv2d(
                self.input_channels,
                hidden_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                padding=(0, 0),
                bias=not pre_norm,
            )
        elif stem_type == "hmlp":
            assert patch_size == 16, "The hMLP stem requires patch_size=16"

            if stem_norm_layer_type is None:
                self.conv_proj = hMLPStem(self.input_channels, hidden_dim)
            elif stem_norm_layer_type == "BatchNorm2d":
                self.conv_proj = hMLPStem(self.input_channels, hidden_dim, norm_layer=nn.BatchNorm2d)
            elif stem_norm_layer_type == "LayerNorm2d":
                self.conv_proj = hMLPStem(self.input_channels, hidden_dim, norm_layer=LayerNorm2d)
            else:
                raise ValueError(f"Unknown stem_norm_layer_type '{stem_norm_layer_type}'")
        else:
            raise ValueError(f"Unknown stem_type '{stem_type}'")

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
        elif mlp_layer_type == "Norm_SwiGLU_FFN":
            mlp_layer = partial(SwiGLU_FFN, norm_layer=norm_layer, norm_eps=norm_layer_eps)  # type: ignore[assignment]
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown mlp_layer_type '{mlp_layer_type}'")

        if act_layer_type is not None:
            act_layer = get_activation_module(act_layer_type)

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.abs_pos_embed = abs_pos_embed
        self.pos_embed_special_tokens = pos_embed_special_tokens
        self.pos_embed_interpolation_mode = pos_embed_interpolation_mode
        self.pos_embed_antialias = pos_embed_antialias
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_reg_tokens = num_reg_tokens
        self.attn_pool_special_tokens = attn_pool_special_tokens
        self.mlp_head = mlp_head
        dpr = stochastic_depth_rates(drop_path_rate, num_layers)

        self.patch_embed = PatchEmbed()

        seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.num_special_tokens = 0
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
        if self.abs_pos_embed is True:
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        else:
            self.pos_embedding = None

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            moe_layers,
            dropout,
            attention_dropout,
            projection_dropout,
            dpr,
            num_special_tokens=self.num_special_tokens,
            pre_norm=pre_norm,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_norm=attn_norm,
            activation_layer=act_layer,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
            mlp_layer=mlp_layer,
            moe_ffn_type=moe_ffn_type,
            moe_expert_width=moe_expert_width,
            moe_ffn_bias=moe_ffn_bias,
            moe_dropout=moe_dropout,
            moe_num_experts=moe_num_experts,
            moe_num_shared_experts=moe_num_shared_experts,
            moe_num_special_token_experts=moe_num_special_token_experts,
            moe_routed_scaling_factor=moe_routed_scaling_factor,
            moe_routing_type=moe_routing_type,
            moe_top_k=moe_top_k,
            router_bias_update_speed=router_bias_update_speed,
            moe_expert_choice_capacity_factor=moe_expert_choice_capacity_factor,
            moe_capacity_factor=moe_capacity_factor,
            moe_eval_capacity_factor=moe_eval_capacity_factor,
            moe_capacity_multiple_of=moe_capacity_multiple_of,
            router_noise_std=router_noise_std,
            router_g_shard_loss_weight=router_g_shard_loss_weight,
            router_importance_loss_weight=router_importance_loss_weight,
            router_load_loss_weight=router_load_loss_weight,
        )
        self.moe_spec = self.encoder.moe_spec

        if post_norm is True and norm_after_pool is False:
            self.norm = norm_layer(hidden_dim, eps=norm_layer_eps)
        else:
            self.norm = nn.Identity()

        if post_norm is True and norm_after_pool is True:
            self.embedding_norm = norm_layer(hidden_dim, eps=norm_layer_eps)
        else:
            self.embedding_norm = nn.Identity()

        if attn_pool_head is False:
            self.attn_pool = None
        else:
            if attn_pool_type == "MultiHeadAttentionPool":
                attn_pool = MultiHeadAttentionPool
                if attn_pool_num_heads is None:
                    attn_pool_num_heads = num_heads
            elif attn_pool_type == "EfficientProbing":
                attn_pool = EfficientProbing
                if attn_pool_num_heads is None:
                    attn_pool_num_heads = 1
            else:
                raise ValueError(f"Unknown attn_pool_type '{attn_pool_type}'")

            self.attn_pool = attn_pool(
                hidden_dim,
                attn_pool_num_heads,
                mlp_dim,
                qkv_bias=True,
                norm_eps=attn_pool_norm_eps,
                activation_layer=get_activation_module(attn_pool_act_layer_type),
            )

        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = hidden_dim
        self.feature_dim = hidden_dim
        self.decoder_block = partial(
            ViTEncoderBlock,
            16,
            mlp_dim=None,
            dropout=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            drop_path=0.0,
            activation_layer=act_layer,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
            mlp_layer=mlp_layer,
        )

        # Weight initialization
        if isinstance(self.conv_proj, nn.Conv2d):
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        head_bias_init = -math.log(num_classes) if num_classes > 0 else 0.0
        if isinstance(self.classifier, nn.Linear):
            nn.init.zeros_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, head_bias_init)
        elif isinstance(self.classifier, nn.Sequential) and isinstance(self.classifier[-1], nn.Linear):
            nn.init.zeros_(self.classifier[-1].weight)
            if self.classifier[-1].bias is not None:
                nn.init.constant_(self.classifier[-1].bias, head_bias_init)

    def _get_pos_embed(self, H: int, W: int) -> Optional[torch.Tensor]:
        if self.pos_embedding is None:
            return None

        if self.dynamic_size is False:
            return self.pos_embedding

        if H == self.size[0] and W == self.size[1]:
            return self.pos_embedding

        return adjust_position_embedding(
            self.pos_embedding,
            (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
            (H // self.patch_size, W // self.patch_size),
            self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            interpolation_mode=self.pos_embed_interpolation_mode,
            antialias=self.pos_embed_antialias,
        )

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad_(True)

        if unfreeze_features is True:
            for param in self.norm.parameters():
                param.requires_grad_(True)
            for param in self.embedding_norm.parameters():
                param.requires_grad_(True)
            if self.attn_pool is not None:
                for param in self.attn_pool.parameters():
                    param.requires_grad_(True)

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        self.encoder.set_grad_checkpointing(
            enable=enable,
            segments=segments,
            preserve_rng_state=preserve_rng_state,
            use_reentrant=use_reentrant,
        )

    def update_moe_expert_biases(self, expert_loads: torch.Tensor) -> None:
        self.encoder.update_moe_expert_biases(expert_loads)

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.encoder.set_causal_attention(is_causal)

    def strip_for_forward_features(self) -> None:
        super().strip_for_forward_features()
        self.embedding_norm = nn.Identity()
        self.attn_pool = None

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
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

    def masked_encoding_omission(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.Tensor] = None,
        return_all_features: bool = False,
        return_keys: Literal["all", "tokens", "embedding"] = "tokens",
        *,
        return_moe_training_output: bool = False,
    ) -> TokenOmissionResultType:
        H, W = x.shape[-2:]

        x = self.conv_proj(x)
        x = self.patch_embed(x)

        pos_embedding = self._get_pos_embed(H, W)
        if pos_embedding is not None:
            if self.pos_embed_special_tokens is True:
                x = x + pos_embedding[:, self.num_special_tokens :, :]
            else:
                x = x + pos_embedding

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

        # Expand special tokens to batch size and prepend in order [REG..., CLS, PATCH...]
        special_tokens: list[torch.Tensor] = []
        if self.reg_tokens is not None:
            if pos_embedding is not None and self.pos_embed_special_tokens is True:
                reg_tokens = self.reg_tokens + pos_embedding[:, 0 : self.num_reg_tokens, :]
            else:
                reg_tokens = self.reg_tokens

            special_tokens.append(reg_tokens.expand(x.size(0), -1, -1))

        if self.class_token is not None:
            if pos_embedding is not None and self.pos_embed_special_tokens is True:
                cls_token = self.class_token + pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
            else:
                cls_token = self.class_token

            special_tokens.append(cls_token.expand(x.size(0), -1, -1))

        if len(special_tokens) > 0:
            x = torch.concat(special_tokens + [x], dim=1)

        moe_training_output: Optional[MoETrainingOutputType] = None
        if return_all_features is True:
            if return_moe_training_output is True:
                xs, moe_training_output = self.encoder.forward_features(x, return_moe_training_output=True)
            else:
                xs = self.encoder.forward_features(x)

            xs[-1] = self.norm(xs[-1])
            x = torch.stack(xs, dim=-1)
        else:
            if return_moe_training_output is True:
                x, moe_training_output = self.encoder(x, return_moe_training_output=True)
            else:
                x = self.encoder(x)

            x = self.norm(x)

        result: TokenOmissionResultType = {}
        if return_keys in ("all", "tokens"):
            result["tokens"] = x

        if return_keys in ("all", "embedding"):
            if return_all_features is True:
                x = x[..., -1]

            result["embedding"] = self.embedding_from_features(x)

        if moe_training_output is not None:
            result["moe_training_output"] = moe_training_output

        return result

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
        *,
        return_moe_training_output: bool = False,
    ) -> TokenRetentionResultType:
        H, W = x.shape[-2:]

        x = self.conv_proj(x)
        x = mask_tensor(x, mask, mask_token=mask_token, patch_factor=self.max_stride // self.stem_stride)

        x = self.patch_embed(x)
        pos_embedding = self._get_pos_embed(H, W)

        if pos_embedding is not None and self.pos_embed_special_tokens is False:
            x = x + pos_embedding

        # Expand special tokens to batch size and prepend in order [REG..., CLS, PATCH...]
        special_tokens: list[torch.Tensor] = []
        if self.reg_tokens is not None:
            special_tokens.append(self.reg_tokens.expand(x.size(0), -1, -1))
        if self.class_token is not None:
            special_tokens.append(self.class_token.expand(x.size(0), -1, -1))
        if len(special_tokens) > 0:
            x = torch.concat(special_tokens + [x], dim=1)

        if pos_embedding is not None and self.pos_embed_special_tokens is True:
            x = x + pos_embedding

        moe_training_output: Optional[MoETrainingOutputType] = None
        if return_moe_training_output is True:
            x, moe_training_output = self.encoder(x, return_moe_training_output=True)
        else:
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
            result["embedding"] = self.embedding_from_features(x)

        if moe_training_output is not None:
            result["moe_training_output"] = moe_training_output

        return result

    def forward_features(
        self,
        x: torch.Tensor,
        return_input_embedding: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        H, W = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)
        patch_embedding = x
        pos_embedding = self._get_pos_embed(H, W)

        if pos_embedding is not None and self.pos_embed_special_tokens is False:
            x = x + pos_embedding

        # Expand special tokens to batch size and prepend in order [REG..., CLS, PATCH...]
        special_tokens: list[torch.Tensor] = []
        if self.reg_tokens is not None:
            special_tokens.append(self.reg_tokens.expand(x.size(0), -1, -1))
        if self.class_token is not None:
            special_tokens.append(self.class_token.expand(x.size(0), -1, -1))
        if len(special_tokens) > 0:
            x = torch.concat(special_tokens + [x], dim=1)

        if return_input_embedding is True:
            if len(special_tokens) > 0:
                input_embedding = torch.concat(special_tokens + [patch_embedding], dim=1)
            else:
                input_embedding = patch_embedding
        else:
            input_embedding = None  # For TorchScript compatibility

        if pos_embedding is not None and self.pos_embed_special_tokens is True:
            x = x + pos_embedding

        if return_moe_training_output is True:
            x, moe_training_output = self.encoder(x, attn_mask=attn_mask, return_moe_training_output=True)
            x = self.norm(x)
            if return_input_embedding is True and input_embedding is not None:
                x = torch.stack([input_embedding, x], dim=-1)

            return (x, moe_training_output)

        x = self.encoder(x, attn_mask=attn_mask)
        x = self.norm(x)
        if return_input_embedding is True and input_embedding is not None:
            return torch.stack([input_embedding, x], dim=-1)

        return x

    def flatten_features(self, features: torch.Tensor, include_special_tokens: bool = True) -> torch.Tensor:
        if include_special_tokens is False:
            return features[:, self.num_special_tokens :]

        return features

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.embedding_norm(self._pool(features))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.embedding_from_features(x)

    def forward(
        self, x: torch.Tensor, *, return_moe_training_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        if return_moe_training_output is True:
            x, moe_training_output = self.forward_features(x, return_moe_training_output=True)
            x = self.embedding_from_features(x)
            return (self.classify(x), moe_training_output)

        return self.classify(self.embedding(x))

    def create_classifier(
        self, embed_dim: Optional[int] = None, head_bias: Optional[bool] = None, mlp_head: Optional[bool] = None
    ) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size
        if head_bias is None:
            head_bias = self.head_bias
        if mlp_head is None:
            mlp_head = self.mlp_head

        if mlp_head is True:
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, self.num_classes, bias=head_bias),
            )

        return nn.Linear(embed_dim, self.num_classes, bias=head_bias)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        old_size = self.size
        super().adjust_size(new_size)
        if self.pos_embedding is None:
            return

        with torch.no_grad():
            pos_embedding = adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
                interpolation_mode=self.pos_embed_interpolation_mode,
            )

        self.pos_embedding = nn.Parameter(pos_embedding)


# Sparse MoE Vision Transformer Model Naming Convention
# =====================================================
#
# Model names follow a structured pattern to encode architectural choices:
# vit_[v]moe_[size][patch_size]_[components]
#
# Core Components:
# - vit_moe_    : Vision Transformer with MoE FFN layers
# - vit_vmoe_   : Vision Transformer with V-MoE FFN layers
# - size        : Model size (s=small, b=base, l=large, or specific like so150m)
# - patch_size  : Patch size (e.g., 16, 32 for 16x16, 32x32 patches)
#
# Sparse MoE Components:
# - {N}e        : Number of total experts, e.g. 8e for V-MoE
# - {S}s        : Number of shared experts included in the total, e.g. 32e1s has 31 routed and 1 shared
# - {P}p        : Number of prefix experts included in the total, e.g. 32e1s1p has 30 routed, 1 shared, and 1 prefix
# - {K}k        : Token-choice routing with top-k K, e.g. 2k
# - {C}c        : Expert-choice routing with capacity factor C, e.g. 2c
# - every{N}    : MoE FFN every N transformer blocks, matching upstream "Every N"
# - last{N}     : MoE FFN in each of the last N transformer blocks
# - last{N}s{S} : Last N MoE FFN layers, separated by stride S and ending at the final block

registry.register_model_config(
    "vit_moe_t16_4e1s1p_2k_last1",
    ViT_MoE,
    config={
        "patch_size": 16,
        **TINY,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 384,
        "moe_num_experts": 4,
        "moe_num_shared_experts": 1,
        "moe_num_special_token_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_last_n_layers": 1,
    },
)
registry.register_model_config(
    "vit_moe_t16_4e1s_2c_last1_avg",
    ViT_MoE,
    config={
        "patch_size": 16,
        **TINY,
        "class_token": False,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 192,
        "moe_num_experts": 4,
        "moe_num_shared_experts": 1,
        "moe_routing_type": "expert_choice",
        "moe_expert_choice_capacity_factor": 2.0,
        "moe_last_n_layers": 1,
    },
)
registry.register_model_config(
    "vit_moe_m16_32e1s_2k_every2_ls_avg",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "class_token": False,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    },
)
registry.register_model_config(
    "vit_moe_m16_32e1s_2c_every2_ls_avg",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "class_token": False,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routing_type": "expert_choice",
        "moe_expert_choice_capacity_factor": 2.0,
        "moe_every_n_layers": 2,
    },
)
registry.register_model_config(
    "vit_moe_b16_32e1s_2k_every2_avg",
    ViT_MoE,
    config={
        "patch_size": 16,
        **BASE,
        "class_token": False,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 640,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    },
)
registry.register_model_config(
    "vit_moe_b16_32e1s_2k_last8_ap",
    ViT_MoE,
    config={
        "patch_size": 16,
        **BASE,
        "class_token": False,
        "attn_pool_head": True,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 640,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_last_n_layers": 8,
    },
)
registry.register_model_config(
    "vit_moe_so150m_p16_32e1s_2k_last12_avg",
    ViT_MoE,
    config={
        "patch_size": 16,
        **SO150,
        "class_token": False,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 768,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_last_n_layers": 12,
    },
)
registry.register_model_config(
    "vit_moe_l16_32e1s_2k_every2_avg",
    ViT_MoE,
    config={
        "patch_size": 16,
        **LARGE,
        "class_token": False,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 896,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    },
)

# With registers
####################

registry.register_model_config(
    "vit_moe_reg1_m16_32e1s_2c_last6_ls_ap",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "num_reg_tokens": 1,
        "class_token": False,
        "attn_pool_head": True,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routing_type": "expert_choice",
        "moe_expert_choice_capacity_factor": 2.0,
        "moe_last_n_layers": 6,
    },
)
registry.register_model_config(
    "vit_moe_reg1_m16_32e1s_2k_last8_ls_ap",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "num_reg_tokens": 1,
        "class_token": False,
        "attn_pool_head": True,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_last_n_layers": 8,
    },
)
registry.register_model_config(
    "vit_moe_reg1_m16_d14_32e1s_3k_last8_ls",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "num_layers": 14,
        "num_reg_tokens": 1,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 384,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.75,
        "moe_top_k": 3,
        "moe_last_n_layers": 8,
    },
)
registry.register_model_config(
    "vit_moe_reg4_m16_32e1s1p_2c_last6_ls",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "num_reg_tokens": 4,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_num_special_token_experts": 1,
        "moe_routing_type": "expert_choice",
        "moe_expert_choice_capacity_factor": 2.0,
        "moe_last_n_layers": 6,
    },
)
registry.register_model_config(
    "vit_moe_reg1_b16_32e1s_2k_every2_ls",
    ViT_MoE,
    config={
        "patch_size": 16,
        **BASE,
        "num_reg_tokens": 1,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 640,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    },
)
registry.register_model_config(
    "vit_moe_reg1_b16_32e1s_3k_last6_ls_ap",
    ViT_MoE,
    config={
        "patch_size": 16,
        **BASE,
        "num_reg_tokens": 1,
        "class_token": False,
        "attn_pool_head": True,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.75,
        "moe_top_k": 3,
        "moe_last_n_layers": 6,
    },
)
registry.register_model_config(
    "vit_moe_reg1_b16_d14_32e1s_2k_every2_ls",
    ViT_MoE,
    config={
        "patch_size": 16,
        **BASE,
        "num_layers": 14,
        "num_reg_tokens": 1,
        "layer_scale_init_value": 1e-5,
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 640,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.5,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    },
)
registry.register_model_config(
    "vit_moe_reg1_b16_d14_32e1s_3k_last10_swiglu_ls_ap",
    ViT_MoE,
    config={
        "patch_size": 16,
        **BASE,
        "num_layers": 14,
        "num_reg_tokens": 1,
        "class_token": False,
        "attn_pool_head": True,
        "layer_scale_init_value": 1e-5,
        "mlp_layer_type": "SwiGLU_FFN",
        "moe_ffn_type": "MoE_FFN",
        "moe_expert_width": 512,
        "moe_num_experts": 32,
        "moe_num_shared_experts": 1,
        "moe_routed_scaling_factor": 1.75,
        "moe_top_k": 3,
        "moe_last_n_layers": 10,
    },
)

# V-MoE
###########################

registry.register_model_config(
    "vit_vmoe_vs32_8e_2k_last2s2",
    ViT_MoE,
    config={
        "patch_size": 32,
        **V_MOE_SMALL,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 2,
        "moe_last_n_layers_stride": 2,
        "mlp_head": True,
    },
)
registry.register_model_config(
    "vit_vmoe_s16_8e_2k_last3s2",
    ViT_MoE,
    config={
        "patch_size": 16,
        **SMALL,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 3,
        "moe_last_n_layers_stride": 2,
    },
)
registry.register_model_config(
    "vit_vmoe_m16_16e_2k_every2_ls",
    ViT_MoE,
    config={
        "patch_size": 16,
        **MEDIUM,
        "layer_scale_init_value": 1e-5,
        "moe_num_experts": 16,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    },
)
registry.register_model_config(
    "vit_vmoe_b16_8e_2k_every2",
    ViT_MoE,
    config={"patch_size": 16, **BASE, "moe_num_experts": 8, "moe_top_k": 2, "moe_every_n_layers": 2},
)
registry.register_model_config(
    "vit_vmoe_b16_16e_2k_every2",
    ViT_MoE,
    config={"patch_size": 16, **BASE, "moe_num_experts": 16, "moe_top_k": 2, "moe_every_n_layers": 2},
)
registry.register_model_config(
    "vit_vmoe_so150m_p16_16e_2k_last4s2",
    ViT_MoE,
    config={
        "patch_size": 16,
        **SO150,
        "moe_num_experts": 16,
        "moe_top_k": 2,
        "moe_last_n_layers": 4,
        "moe_last_n_layers_stride": 2,
    },
)

# With registers
####################

registry.register_model_config(
    "vit_vmoe_reg1_vs32_8e_2k_last2s2",
    ViT_MoE,
    config={
        "patch_size": 32,
        **V_MOE_SMALL,
        "num_reg_tokens": 1,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 2,
        "moe_last_n_layers_stride": 2,
        "mlp_head": True,
    },
)
registry.register_model_config(
    "vit_vmoe_reg1_s16_8e_2k_last3s2",
    ViT_MoE,
    config={
        "patch_size": 16,
        **SMALL,
        "num_reg_tokens": 1,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 3,
        "moe_last_n_layers_stride": 2,
    },
)
registry.register_model_config(
    "vit_vmoe_reg8_so150m_p16_16e_2k_last3s2",
    ViT_MoE,
    config={
        "patch_size": 16,
        **SO150,
        "num_reg_tokens": 8,
        "moe_num_experts": 16,
        "moe_top_k": 2,
        "moe_last_n_layers": 3,
        "moe_last_n_layers_stride": 2,
    },
)
