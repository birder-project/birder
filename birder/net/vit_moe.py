"""
V-MoE, adapted from https://github.com/google-research/vmoe

Paper "Scaling Vision with Sparse Mixture of Experts", https://arxiv.org/abs/2106.05974
"""

# Reference license: Apache-2.0

import copy
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
from birder.layers.activations import get_activation_module
from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import SMALL
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import TokenRetentionResultType
from birder.net.vit import Attention
from birder.net.vit import EncoderBlock as ViTEncoderBlock
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding

AuxLossesType = dict[str, torch.Tensor]

V_MOE_TRAIN_IMAGES_PER_GROUP = 8
V_MOE_EVAL_IMAGES_PER_GROUP = 1
V_MOE_SMALL = {"num_layers": 8, "num_heads": 8, "hidden_dim": 512, "mlp_dim": 2048, "drop_path_rate": 0.0}


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _resolve_moe_layers(
    num_layers: int,
    *,
    moe_layers: Optional[list[int]],
    moe_every_n_layers: Optional[int],
    moe_last_n_layers: Optional[int],
) -> list[int]:
    assert (
        sum(x is not None for x in (moe_layers, moe_every_n_layers, moe_last_n_layers)) == 1
    ), "Exactly one of moe_layers, moe_every_n_layers, or moe_last_n_layers must be set"

    if moe_layers is not None:
        return moe_layers

    if moe_every_n_layers is not None:
        return list(range(moe_every_n_layers - 1, num_layers, moe_every_n_layers))

    return list(range(1, num_layers, 2))[-moe_last_n_layers:]  # type: ignore[operator]


def _sum_aux_losses(aux_losses: list[AuxLossesType], ref: torch.Tensor) -> AuxLossesType:
    if len(aux_losses) == 0:
        return {key: ref.new_zeros(()) for key in ("auxiliary_loss", "g_shard_loss", "importance_loss", "load_loss")}

    return {
        key: torch.stack([losses[key] for losses in aux_losses]).sum()
        for key in ("auxiliary_loss", "g_shard_loss", "importance_loss", "load_loss")
    }


def _cv_squared(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6 if x.dtype == torch.float16 else 1e-12

    x = x.float()
    return x.std(dim=-1, correction=0).square() / x.mean(dim=-1).clamp_min(eps).square()


class NoisyTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 1,
        noise_std: float = 1.0,
        capacity_factor: float = 1.05,
        eval_capacity_factor: float = 4.0,
        capacity_multiple_of: Optional[int] = 4,
        g_shard_loss_weight: float = 0.0,
        importance_loss_weight: float = 0.005,
        load_loss_weight: float = 0.005,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.capacity_multiple_of = capacity_multiple_of
        self.g_shard_loss_weight = g_shard_loss_weight
        self.importance_loss_weight = importance_loss_weight
        self.load_loss_weight = load_loss_weight
        self.moe_loss_output = False
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def _capacity(self, group_size: int) -> int:
        capacity_factor = self.capacity_factor if self.training is True else self.eval_capacity_factor
        capacity = math.ceil(group_size * self.top_k * capacity_factor / self.num_experts)
        capacity = max(1, capacity)
        if self.capacity_multiple_of is not None:
            capacity += (-capacity) % self.capacity_multiple_of

        return capacity

    def _g_shard_auxiliary_loss(self, gates: torch.Tensor) -> torch.Tensor:
        mean_gates_per_expert = gates.mean(dim=1)
        top1 = F.one_hot(gates.argmax(dim=-1), num_classes=self.num_experts).to(  # pylint: disable=not-callable
            dtype=gates.dtype
        )
        mean_top1_per_expert = top1.mean(dim=1)
        return (mean_top1_per_expert * mean_gates_per_expert).mean(dim=-1) * (self.num_experts**2)

    def _importance_auxiliary_loss(self, gates: torch.Tensor) -> torch.Tensor:
        importance_per_expert = gates.sum(dim=1)
        return _cv_squared(importance_per_expert)

    def _load_auxiliary_loss(self, logits: torch.Tensor, logits_noisy: torch.Tensor, noise_std: float) -> torch.Tensor:
        threshold_index = logits_noisy.topk(self.top_k, dim=-1).indices[..., -1]
        threshold = logits_noisy.gather(-1, threshold_index.unsqueeze(-1)).squeeze(-1)
        noise_required_to_win = (threshold.unsqueeze(-1) - logits) / noise_std
        p = 0.5 * (1.0 - torch.erf(noise_required_to_win / math.sqrt(2.0)))

        return _cv_squared(p.mean(dim=1))

    def _make_aux_losses(
        self, logits: torch.Tensor, logits_noisy: torch.Tensor, gates: torch.Tensor, noise_std: Optional[float]
    ) -> AuxLossesType:
        g_shard_loss = logits.new_zeros(())
        if self.g_shard_loss_weight > 0.0:
            g_shard_loss = self._g_shard_auxiliary_loss(gates).mean()

        importance_loss = logits.new_zeros(())
        if self.importance_loss_weight > 0.0:
            importance_loss = self._importance_auxiliary_loss(logits.softmax(dim=-1)).mean()

        load_loss = logits.new_zeros(())
        if self.load_loss_weight > 0.0 and noise_std is not None:
            load_loss = self._load_auxiliary_loss(logits, logits_noisy, noise_std).mean()

        auxiliary_loss = (
            self.g_shard_loss_weight * g_shard_loss
            + self.importance_loss_weight * importance_loss
            + self.load_loss_weight * load_loss
        )
        return {
            "auxiliary_loss": auxiliary_loss,
            "g_shard_loss": g_shard_loss,
            "importance_loss": importance_loss,
            "load_loss": load_loss,
        }

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[AuxLossesType]]:
        """
        Route grouped tokens

        Parameters
        ----------
        x
            Tensor of shape (G, S, C).
        """

        G, S, _ = x.size()
        logits = self.gate(x)
        if self.training is True and self.noise_std > 0.0:
            scaled_noise_std: Optional[float] = self.noise_std / self.num_experts
            logits_noisy = logits + torch.randn_like(logits) * scaled_noise_std
        else:
            scaled_noise_std = None
            logits_noisy = logits

        gates = logits_noisy.softmax(dim=-1)
        combine_weights, expert_index = gates.topk(self.top_k, dim=-1)

        # Match the V-MoE "vanilla" priority rule: all top-1 choices get
        # capacity before top-2 choices, and so on.
        expert_index_flat = expert_index.permute(0, 2, 1).reshape(G, S * self.top_k)
        expert_one_hot = F.one_hot(expert_index_flat, num_classes=self.num_experts).to(  # pylint: disable=not-callable
            dtype=torch.int64
        )
        buffer_index = torch.cumsum(expert_one_hot, dim=1) * expert_one_hot - 1
        buffer_index = buffer_index.reshape(G, self.top_k, S, self.num_experts).permute(0, 2, 1, 3)
        buffer_index = buffer_index.gather(-1, expert_index.unsqueeze(-1)).squeeze(-1)

        capacity = self._capacity(S)
        valid_mask = buffer_index < capacity

        if self.training is True and self.moe_loss_output is True:
            aux_losses = self._make_aux_losses(logits, logits_noisy, gates, scaled_noise_std)
        else:
            aux_losses = None

        return (expert_index, buffer_index, combine_weights * valid_mask.to(combine_weights.dtype), aux_losses)

    def set_moe_loss_output(self, enable: bool = True) -> None:
        self.moe_loss_output = enable

    def forward(
        self, x: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, AuxLossesType]
    ):
        expert_index, buffer_index, combine_weights, aux_losses = self._route(x)
        if aux_losses is not None:
            return (expert_index, buffer_index, combine_weights, aux_losses)

        return (expert_index, buffer_index, combine_weights)


class SparseMoE_FFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
        dropout: float = 0.0,
        num_experts: int = 8,
        top_k: int = 1,
        capacity_factor: float = 1.05,
        eval_capacity_factor: float = 4.0,
        capacity_multiple_of: Optional[int] = 4,
        router_noise_std: float = 1.0,
        router_g_shard_loss_weight: float = 0.0,
        router_importance_loss_weight: float = 0.005,
        router_load_loss_weight: float = 0.005,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_loss_output = False
        self.router = NoisyTopKRouter(
            in_features,
            num_experts,
            top_k=top_k,
            noise_std=router_noise_std,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            capacity_multiple_of=capacity_multiple_of,
            g_shard_loss_weight=router_g_shard_loss_weight,
            importance_loss_weight=router_importance_loss_weight,
            load_loss_weight=router_load_loss_weight,
        )
        expert = FFN(in_features, hidden_features, act_layer=act_layer, bias=bias, dropout=dropout)
        self.experts = _get_clones(expert, num_experts)

    def _group_size(self, seq_length: int) -> int:
        images_per_group = V_MOE_TRAIN_IMAGES_PER_GROUP if self.training is True else V_MOE_EVAL_IMAGES_PER_GROUP
        return images_per_group * seq_length

    def _group_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_tokens = x.size(0) * x.size(1)
        group_size = self._group_size(x.size(1))
        pad_tokens = (-num_tokens) % group_size

        x = x.reshape(num_tokens, x.size(-1))
        if pad_tokens > 0:
            x = F.pad(x, (0, 0, 0, pad_tokens))

        x = x.reshape(-1, group_size, x.size(-1))
        return (x, pad_tokens)

    def set_moe_loss_output(self, enable: bool = True) -> None:
        self.moe_loss_output = enable
        self.router.set_moe_loss_output(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, AuxLossesType]:
        B, N, C = x.size()
        grouped_x, pad_tokens = self._group_tokens(x)
        G, S, _ = grouped_x.size()
        capacity = self.router._capacity(S)

        if self.training is True and self.moe_loss_output is True:
            expert_index, buffer_index, combine_weights, aux_losses = self.router(grouped_x)
        else:
            expert_index, buffer_index, combine_weights = self.router(grouped_x)
            aux_losses = None

        assignment_x = grouped_x.unsqueeze(2).expand(-1, -1, self.top_k, -1)
        valid_mask = combine_weights > 0
        expert_slot = torch.arange(G, device=x.device).view(G, 1, 1) * capacity + buffer_index
        linear_index = expert_index * (G * capacity) + expert_slot
        linear_index = torch.where(valid_mask, linear_index, torch.zeros_like(linear_index))

        flat_index = linear_index.reshape(-1)
        flat_values = assignment_x.reshape(-1, C) * valid_mask.reshape(-1, 1).to(dtype=x.dtype)
        expert_inputs = grouped_x.new_zeros((self.num_experts * G * capacity, C))
        expert_inputs.scatter_add_(0, flat_index.unsqueeze(-1).expand(-1, C), flat_values)
        expert_inputs = expert_inputs.reshape(self.num_experts, G * capacity, C)

        expert_outputs = torch.stack([expert(expert_inputs[idx]) for idx, expert in enumerate(self.experts)], dim=0)
        expert_outputs = expert_outputs.reshape(self.num_experts * G * capacity, C)
        combined = expert_outputs[flat_index].reshape(G, S, self.top_k, C)
        combined = combined * combine_weights.unsqueeze(-1)
        x = combined.sum(dim=2).reshape(-1, C)
        if pad_tokens > 0:
            x = x[:-pad_tokens]

        x = x.reshape(B, N, C)
        if self.training is True and self.moe_loss_output is True:
            return (x, aux_losses)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        activation_layer: Callable[..., nn.Module],
        layer_scale_init_value: Optional[float] = None,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=norm_layer_eps)
        self.attn = Attention(hidden_dim, num_heads=num_heads, attn_drop=attention_dropout, proj_drop=0.0)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_dim, eps=norm_layer_eps)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=dropout)
        self.moe_loss_output = False
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def set_moe_loss_output(self, enable: bool = True) -> None:
        if isinstance(self.mlp, SparseMoE_FFN):
            self.moe_loss_output = enable
            self.mlp.set_moe_loss_output(enable)
        else:
            self.moe_loss_output = False

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, AuxLossesType]:
        attn_out, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(self.layer_scale_1(attn_out))
        x_norm = self.norm2(x)
        if self.training is True and self.moe_loss_output is True:
            y, aux_losses = self.mlp(x_norm)
            x = x + self.drop_path(self.layer_scale_2(y))
            return (x, aux_losses)

        mlp_out = self.mlp(x_norm)
        x = x + self.drop_path(self.layer_scale_2(mlp_out))

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
        dpr: list[float],
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer_eps: float = 1e-6,
        moe_dropout: float = 0.0,
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.05,
        moe_eval_capacity_factor: float = 4.0,
        moe_capacity_multiple_of: Optional[int] = 4,
        router_noise_std: float = 1.0,
        router_g_shard_loss_weight: float = 0.0,
        router_importance_loss_weight: float = 0.005,
        router_load_loss_weight: float = 0.005,
    ) -> None:
        super().__init__()
        pre_layers = []
        if dropout > 0.0:
            pre_layers.append(nn.Dropout(dropout))

        self.pre_block = nn.Sequential(*pre_layers)
        layers = []
        for i in range(num_layers):
            if i in moe_layers:
                mlp_dropout = moe_dropout
                mlp_layer: Callable[..., nn.Module] = partial(
                    SparseMoE_FFN,
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
            else:
                mlp_dropout = dropout
                mlp_layer = FFN

            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    mlp_dropout,
                    attention_dropout,
                    dpr[i],
                    activation_layer=activation_layer,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer_eps=norm_layer_eps,
                    mlp_layer=mlp_layer,
                )
            )

        self.block = nn.ModuleList(layers)
        self.moe_loss_output = False

    def set_moe_loss_output(self, enable: bool = True) -> None:
        self.moe_loss_output = enable
        for blk in self.block:
            blk.set_moe_loss_output(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, AuxLossesType]:
        x = self.pre_block(x)
        aux_loss_list: list[AuxLossesType] = []
        for blk in self.block:
            if self.training is True and blk.moe_loss_output is True:
                x, aux_losses = blk(x)
                if aux_losses is not None:
                    aux_loss_list.append(aux_losses)
            else:
                x = blk(x)

        if self.training is True and self.moe_loss_output is True:
            return (x, _sum_aux_losses(aux_loss_list, x))

        return x

    def forward_features(
        self,
        x: torch.Tensor,
        out_indices: Optional[list[int]] = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], AuxLossesType]:
        x = self.pre_block(x)

        out_indices_set = set(out_indices) if out_indices is not None else None
        xs = []
        aux_loss_list: list[AuxLossesType] = []
        for idx, blk in enumerate(self.block):
            if self.training is True and blk.moe_loss_output is True:
                x, aux_losses = blk(x)
                if aux_losses is not None:
                    aux_loss_list.append(aux_losses)
            else:
                x = blk(x)

            if out_indices_set is None or idx in out_indices_set:
                xs.append(x)

        if self.training is True and self.moe_loss_output is True:
            return (xs, _sum_aux_losses(aux_loss_list, x))

        return xs


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
        attention_dropout = 0.0
        dropout = 0.0
        pos_embed_special_tokens: bool = self.config.get("pos_embed_special_tokens", True)
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-6)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        moe_layers = _resolve_moe_layers(
            num_layers,
            moe_layers=self.config.get("moe_layers", None),
            moe_every_n_layers=self.config.get("moe_every_n_layers", None),
            moe_last_n_layers=self.config.get("moe_last_n_layers", None),
        )
        moe_dropout: float = self.config.get("moe_dropout", 0.0)
        moe_num_experts: int = self.config.get("moe_num_experts", 8)
        moe_top_k: int = self.config.get("moe_top_k", 1)
        moe_capacity_factor: float = self.config.get("moe_capacity_factor", 1.05)
        moe_eval_capacity_factor: float = self.config.get("moe_eval_capacity_factor", 4.0)
        moe_capacity_multiple_of: Optional[int] = self.config.get("moe_capacity_multiple_of", 4)
        router_noise_std: float = self.config.get("router_noise_std", 1.0)
        router_g_shard_loss_weight: float = self.config.get("router_g_shard_loss_weight", 0.0)
        router_importance_loss_weight: float = self.config.get("router_importance_loss_weight", 0.005)
        router_load_loss_weight: float = self.config.get("router_load_loss_weight", 0.005)
        mlp_head = self.config.get("mlp_head", False)
        act_layer_type: str = self.config.get("act_layer_type", "gelu_tanh")
        drop_path_rate: float = self.config["drop_path_rate"]

        act_layer = get_activation_module(act_layer_type)

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.pos_embed_special_tokens = pos_embed_special_tokens
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_reg_tokens = num_reg_tokens
        self.mlp_head = mlp_head
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.conv_proj = nn.Conv2d(
            self.input_channels,
            hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        )
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

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            moe_layers,
            dropout,
            attention_dropout,
            dpr,
            activation_layer=act_layer,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer_eps=norm_layer_eps,
            moe_dropout=moe_dropout,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_capacity_factor=moe_capacity_factor,
            moe_eval_capacity_factor=moe_eval_capacity_factor,
            moe_capacity_multiple_of=moe_capacity_multiple_of,
            router_noise_std=router_noise_std,
            router_g_shard_loss_weight=router_g_shard_loss_weight,
            router_importance_loss_weight=router_importance_loss_weight,
            router_load_loss_weight=router_load_loss_weight,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=norm_layer_eps)

        self.embedding_size = hidden_dim
        self.feature_dim = hidden_dim
        self.classifier = self.create_classifier()
        self.moe_loss_output = False
        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = hidden_dim
        self.decoder_block = partial(
            ViTEncoderBlock,
            16,
            mlp_dim=None,
            dropout=0,
            attention_dropout=0,
            drop_path=0,
            activation_layer=act_layer,
            norm_layer_eps=norm_layer_eps,
        )

        # Weight initialization
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

    def set_moe_loss_output(self, enable: bool = True) -> None:
        self.moe_loss_output = enable
        self.encoder.set_moe_loss_output(enable)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.class_token is None:
            return x[:, self.num_special_tokens :].mean(dim=1)

        return x[:, self.num_reg_tokens]

    def masked_encoding_omission(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.Tensor] = None,
        return_all_features: bool = False,
        return_keys: Literal["all", "tokens", "embedding"] = "tokens",
    ) -> TokenOmissionResultType:
        H, W = x.shape[-2:]

        x = self.conv_proj(x)
        x = self.patch_embed(x)

        pos_embedding = self._get_pos_embed(H, W)
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
            if self.pos_embed_special_tokens is True:
                reg_tokens = self.reg_tokens + pos_embedding[:, 0 : self.num_reg_tokens, :]
            else:
                reg_tokens = self.reg_tokens

            special_tokens.append(reg_tokens.expand(x.size(0), -1, -1))

        if self.class_token is not None:
            if self.pos_embed_special_tokens is True:
                cls_token = self.class_token + pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
            else:
                cls_token = self.class_token

            special_tokens.append(cls_token.expand(x.size(0), -1, -1))

        if len(special_tokens) > 0:
            x = torch.concat(special_tokens + [x], dim=1)

        aux_losses: Optional[AuxLossesType] = None
        if return_all_features is True:
            if self.training is True and self.moe_loss_output is True:
                xs, aux_losses = self.encoder.forward_features(x)
            else:
                xs = self.encoder.forward_features(x)

            xs[-1] = self.norm(xs[-1])
            x = torch.stack(xs, dim=-1)
        else:
            if self.training is True and self.moe_loss_output is True:
                x, aux_losses = self.encoder(x)
            else:
                x = self.encoder(x)

            x = self.norm(x)

        result: TokenOmissionResultType = {}
        if return_keys in ("all", "tokens"):
            result["tokens"] = x

        if return_keys in ("all", "embedding"):
            if return_all_features is True:
                x = x[..., -1]

            result["embedding"] = self._pool(x)

        if aux_losses is not None:
            result["auxiliary_losses"] = aux_losses

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

        x = self.patch_embed(x)
        pos_embedding = self._get_pos_embed(H, W)

        if self.pos_embed_special_tokens is False:
            x = x + pos_embedding

        # Expand special tokens to batch size and prepend in order [REG..., CLS, PATCH...]
        special_tokens: list[torch.Tensor] = []
        if self.reg_tokens is not None:
            special_tokens.append(self.reg_tokens.expand(x.size(0), -1, -1))
        if self.class_token is not None:
            special_tokens.append(self.class_token.expand(x.size(0), -1, -1))
        if len(special_tokens) > 0:
            x = torch.concat(special_tokens + [x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + pos_embedding

        aux_losses: Optional[AuxLossesType] = None
        if self.training is True and self.moe_loss_output is True:
            x, aux_losses = self.encoder(x)
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
            result["embedding"] = self._pool(x)

        if aux_losses is not None:
            result["auxiliary_losses"] = aux_losses

        return result

    def forward_features(
        self,
        x: torch.Tensor,
        return_input_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AuxLossesType]:
        H, W = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)
        patch_embedding = x
        pos_embedding = self._get_pos_embed(H, W)

        if self.pos_embed_special_tokens is False:
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

        if self.pos_embed_special_tokens is True:
            x = x + pos_embedding

        if self.training is True and self.moe_loss_output is True:
            x, aux_losses = self.encoder(x)
            x = self.norm(x)
            if return_input_embedding is True and input_embedding is not None:
                x = torch.stack([input_embedding, x], dim=-1)

            return (x, aux_losses)

        x = self.encoder(x)
        x = self.norm(x)
        if return_input_embedding is True and input_embedding is not None:
            return torch.stack([input_embedding, x], dim=-1)

        return x

    def flatten_features(self, features: torch.Tensor, include_special_tokens: bool = True) -> torch.Tensor:
        if include_special_tokens is False:
            return features[:, self.num_special_tokens :]

        return features

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self._pool(features)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.training is True and self.moe_loss_output is True:
            x, _ = x

        return self.embedding_from_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, AuxLossesType]:
        if self.training is True and self.moe_loss_output is True:
            x, aux_losses = self.forward_features(x)
            return (self.classify(self.embedding_from_features(x)), aux_losses)

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
        with torch.no_grad():
            pos_embedding = adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            )

        self.pos_embedding = nn.Parameter(pos_embedding)


# Sparse MoE Vision Transformer Model Naming Convention
# =====================================================
#
# Model names follow a structured pattern to encode architectural choices:
# vit_moe_[size][patch_size]_[components]
#
# Core Components:
# - vit_moe_    : Vision Transformer with sparse V-MoE FFN layers
# - size        : Model size (s=small, b=base)
# - patch_size  : Patch size (e.g., 16, 32 for 16x16, 32x32 patches)
#
# Sparse MoE Components:
# - {N}e      : Number of experts, e.g. 8e
# - {K}k      : Router top-k, e.g. 2k
# - every{N}  : MoE FFN every N transformer blocks, matching upstream "Every N"
# - last{N}   : Last N odd-indexed MoE FFN candidates, matching upstream "Last N"

registry.register_model_config(
    "vit_moe_vs32_8e_2k_last2",
    ViT_MoE,
    config={
        "patch_size": 32,
        **V_MOE_SMALL,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 2,
        "mlp_head": True,
    },
)
registry.register_model_config(
    "vit_moe_s16_8e_2k_last3",
    ViT_MoE,
    config={"patch_size": 16, **SMALL, "moe_num_experts": 8, "moe_top_k": 2, "moe_last_n_layers": 3},
)
registry.register_model_config(
    "vit_moe_b16_8e_2k_every2",
    ViT_MoE,
    config={"patch_size": 16, **BASE, "moe_num_experts": 8, "moe_top_k": 2, "moe_every_n_layers": 2},
)

# With registers
####################

registry.register_model_config(
    "vit_moe_reg1_vs32_8e_2k_last2",
    ViT_MoE,
    config={
        "patch_size": 32,
        **V_MOE_SMALL,
        "num_reg_tokens": 1,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 2,
        "mlp_head": True,
    },
)
registry.register_model_config(
    "vit_moe_reg1_s16_8e_2k_last3",
    ViT_MoE,
    config={
        "patch_size": 16,
        **SMALL,
        "num_reg_tokens": 1,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_last_n_layers": 3,
    },
)
