import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.layers.ffn import FFN
from birder.layers.ffn import SwiGLU_FFN


@dataclass(frozen=True, slots=True)
class MoESpec:
    has_auxiliary_loss: bool
    requires_expert_bias_update: bool

    @property
    def requires_training_output(self) -> bool:
        return self.has_auxiliary_loss or self.requires_expert_bias_update


MoETrainingOutputType = dict[str, torch.Tensor]

V_MOE_TRAIN_IMAGES_PER_GROUP = 8
V_MOE_EVAL_IMAGES_PER_GROUP = 1


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _empty_moe_training_output(ref: torch.Tensor) -> MoETrainingOutputType:
    return {
        "auxiliary_loss": ref.new_zeros(()),
        "g_shard_loss": ref.new_zeros(()),
        "importance_loss": ref.new_zeros(()),
        "load_loss": ref.new_zeros(()),
        "expert_loads": ref.new_empty((0, 0), dtype=torch.int64),
    }


def _cv_squared(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6 if x.dtype == torch.float16 else 1e-12

    x = x.float()
    return x.std(dim=-1, correction=0).square() / x.mean(dim=-1).clamp_min(eps).square()


def _masked_mean(x: torch.Tensor, token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if token_mask is None:
        return x.mean(dim=1)

    token_mask = token_mask.unsqueeze(-1)
    return (x * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)


class SoftMoE_FFN(nn.Module):
    """
    Paper "From Sparse to Soft Mixtures of Experts", https://arxiv.org/abs/2308.00951
    Adapted from: https://github.com/lucidrains/soft-moe-pytorch
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
        dropout: float = 0.0,
        num_experts: int = 4,
        num_slots: int = 1,
        normalize: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.slot_embeds = nn.Parameter(torch.empty(num_experts, num_slots, in_features).normal_(std=0.02))
        self.experts = nn.ModuleList(
            [
                FFN(in_features, hidden_features, act_layer=act_layer, bias=bias, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        if normalize is True:
            self.token_norm = nn.RMSNorm(in_features, eps=norm_eps)
            self.slot_norm = nn.RMSNorm(in_features, eps=norm_eps)
        else:
            self.token_norm = nn.Identity()
            self.slot_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route_x = self.token_norm(x)
        route_slots = self.slot_norm(self.slot_embeds)
        logits = torch.einsum("b n d, e s d -> b n e s", route_x, route_slots)

        dispatch_weights = F.softmax(logits, dim=1)
        combine_weights = F.softmax(logits.reshape(logits.size(0), logits.size(1), -1), dim=-1)
        combine_weights = combine_weights.reshape_as(logits)

        slots = torch.einsum("b n d, b n e s -> b e s d", x, dispatch_weights)
        expert_out = torch.stack([expert(slots[:, idx]) for idx, expert in enumerate(self.experts)], dim=1)
        x = torch.einsum("b e s d, b n e s -> b n d", expert_out, combine_weights)

        return x


class BaseSparseMoE_FFN(nn.Module):
    """
    Common interface for sparse FFNs that optionally emit MoE training data
    """

    def forward(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None, *, return_moe_training_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        raise NotImplementedError


class NoisyTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
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
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Weight initialization
        normal_pdf_at_two = math.exp(-2.0) / math.sqrt(2.0 * math.pi)
        normal_cdf_delta = math.erf(math.sqrt(2.0))
        truncated_normal_stddev = math.sqrt(1.0 - 4.0 * normal_pdf_at_two / normal_cdf_delta)
        gate_std = math.sqrt(1.0 / self.gate.weight.size(1)) / truncated_normal_stddev
        nn.init.trunc_normal_(self.gate.weight, std=gate_std, a=-2.0 * gate_std, b=2.0 * gate_std)

    def _capacity(self, group_size: int) -> int:
        capacity_factor = self.capacity_factor if self.training is True else self.eval_capacity_factor
        capacity = math.ceil(group_size * self.top_k * capacity_factor / self.num_experts)
        capacity = max(1, capacity)
        if self.capacity_multiple_of is not None:
            capacity += (-capacity) % self.capacity_multiple_of

        return capacity

    def _g_shard_auxiliary_loss(
        self, gates: torch.Tensor, top1_index: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mean_gates_per_expert = _masked_mean(gates, token_mask)
        # Match dispatch even when topk and argmax break ties differently.
        top1 = F.one_hot(top1_index, num_classes=self.num_experts).to(dtype=gates.dtype)  # pylint: disable=not-callable
        mean_top1_per_expert = _masked_mean(top1, token_mask)
        return (mean_top1_per_expert * mean_gates_per_expert).mean(dim=-1) * (self.num_experts**2)

    def _importance_auxiliary_loss(
        self, gates: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if token_mask is not None:
            gates = gates * token_mask.unsqueeze(-1)

        importance_per_expert = gates.sum(dim=1)
        return _cv_squared(importance_per_expert)

    def _load_auxiliary_loss(
        self,
        logits: torch.Tensor,
        logits_noisy: torch.Tensor,
        noise_std: float,
        threshold_index: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        threshold = logits_noisy.gather(-1, threshold_index.unsqueeze(-1)).squeeze(-1)
        noise_required_to_win = (threshold.unsqueeze(-1) - logits) / noise_std
        p = 0.5 * (1.0 - torch.erf(noise_required_to_win / math.sqrt(2.0)))

        return _cv_squared(_masked_mean(p, token_mask))

    def _make_moe_training_output(
        self,
        logits: torch.Tensor,
        logits_noisy: torch.Tensor,
        gates: torch.Tensor,
        noise_std: Optional[float],
        expert_index: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> MoETrainingOutputType:
        g_shard_loss = logits.new_zeros(())
        if self.g_shard_loss_weight > 0.0:
            g_shard_loss = self._g_shard_auxiliary_loss(gates, expert_index[..., 0], token_mask).mean()

        importance_loss = logits.new_zeros(())
        if self.importance_loss_weight > 0.0:
            importance_loss = self._importance_auxiliary_loss(logits.softmax(dim=-1), token_mask).mean()

        load_loss = logits.new_zeros(())
        if self.load_loss_weight > 0.0 and noise_std is not None:
            load_loss = self._load_auxiliary_loss(
                logits, logits_noisy, noise_std, expert_index[..., -1], token_mask
            ).mean()

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
            "expert_loads": logits.new_empty((0, 0), dtype=torch.int64),
        }

    def _route(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor], *, return_moe_training_output: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[MoETrainingOutputType]]:
        """
        Route grouped tokens

        Parameters
        ----------
        x
            Tensor of shape (G, S, C).
        token_mask
            Boolean tensor of shape (G, S), where True marks tokens to route.
        return_moe_training_output
            Whether to calculate and return the router training output.
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
        expert_index = logits_noisy.topk(self.top_k, dim=-1).indices
        combine_weights = gates.gather(-1, expert_index)

        # Match the V-MoE "vanilla" priority rule: all top-1 choices get capacity before top-2 choices, and so on.
        # Use top_k=1 for causal training, later tokens can otherwise displace earlier lower-ranked choices.
        expert_index_flat = expert_index.permute(0, 2, 1).reshape(G, S * self.top_k)
        expert_one_hot = F.one_hot(expert_index_flat, num_classes=self.num_experts)  # pylint: disable=not-callable
        if token_mask is not None:
            expert_one_hot = expert_one_hot.reshape(G, self.top_k, S, self.num_experts)
            expert_one_hot.mul_(token_mask[:, None, :, None])
            expert_one_hot = expert_one_hot.reshape(G, S * self.top_k, self.num_experts)

        buffer_index = torch.cumsum(expert_one_hot, dim=1)
        buffer_index.mul_(expert_one_hot).sub_(1)
        buffer_index = buffer_index.reshape(G, self.top_k, S, self.num_experts).permute(0, 2, 1, 3)
        buffer_index = buffer_index.gather(-1, expert_index.unsqueeze(-1)).squeeze(-1)

        capacity = self._capacity(S)
        valid_mask = buffer_index < capacity
        if token_mask is not None:
            valid_mask = valid_mask & token_mask.unsqueeze(-1)

        if self.training is True and return_moe_training_output is True:
            moe_training_output = self._make_moe_training_output(
                logits, logits_noisy, gates, scaled_noise_std, expert_index, token_mask
            )
        else:
            moe_training_output = None

        return (expert_index, buffer_index, combine_weights * valid_mask, moe_training_output)

    def forward(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None, *, return_moe_training_output: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[MoETrainingOutputType]]:
        expert_index, buffer_index, combine_weights, moe_training_output = self._route(
            x, token_mask, return_moe_training_output=return_moe_training_output
        )
        if return_moe_training_output is True and moe_training_output is None:
            moe_training_output = _empty_moe_training_output(x)

        return (expert_index, buffer_index, combine_weights, moe_training_output)


class VMoE_FFN(BaseSparseMoE_FFN):
    """
    V-MoE, adapted from https://github.com/google-research/vmoe
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
        dropout: float = 0.0,
        num_experts: int = 8,
        top_k: int = 2,
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

        # Weight initialization
        for m in expert.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

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

    def forward(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None, *, return_moe_training_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        B, N, C = x.size()
        grouped_x, pad_tokens = self._group_tokens(x)
        G, S, _ = grouped_x.size()
        capacity = self.router._capacity(S)

        grouped_token_mask = None
        if token_mask is not None:
            grouped_token_mask = token_mask.to(dtype=torch.bool).reshape(B * N)
            if pad_tokens > 0:
                grouped_token_mask = F.pad(grouped_token_mask, (0, pad_tokens))

            grouped_token_mask = grouped_token_mask.reshape(G, S)

        expert_index, buffer_index, combine_weights, moe_training_output = self.router(
            grouped_x, grouped_token_mask, return_moe_training_output=return_moe_training_output
        )

        valid_mask = combine_weights > 0
        expert_slot = torch.arange(G, device=x.device).view(G, 1, 1) * capacity + buffer_index
        linear_index = expert_index * (G * capacity)
        linear_index.add_(expert_slot).mul_(valid_mask)

        flat_index = linear_index.reshape(-1)
        if self.training is True:
            flat_values = (grouped_x.unsqueeze(2) * valid_mask.unsqueeze(-1)).reshape(-1, C)
            expert_inputs = grouped_x.new_zeros((self.num_experts * G * capacity, C))
            expert_inputs.scatter_add_(0, flat_index.unsqueeze(-1).expand(-1, C), flat_values)
            expert_inputs = expert_inputs.reshape(self.num_experts, G * capacity, C)

            expert_outputs = torch.stack([expert(expert_inputs[idx]) for idx, expert in enumerate(self.experts)], dim=0)
            expert_outputs = expert_outputs.reshape(self.num_experts * G * capacity, C)

        else:
            flat_valid = valid_mask.reshape(-1)
            flat_expert_index = expert_index.reshape(-1)
            flat_token_index = torch.arange(G * S, device=x.device).repeat_interleave(self.top_k)
            flat_grouped_x = grouped_x.reshape(G * S, C)

            if torch.is_autocast_enabled(x.device.type) is True:
                expert_dtype = torch.get_autocast_dtype(x.device.type)
            else:
                expert_dtype = grouped_x.dtype

            expert_outputs = grouped_x.new_zeros((self.num_experts * G * capacity, C), dtype=expert_dtype)
            for idx, expert in enumerate(self.experts):
                expert_mask = (flat_expert_index == idx) & flat_valid
                expert_input = flat_grouped_x.index_select(0, flat_token_index[expert_mask])
                expert_output = expert(expert_input)
                expert_outputs.index_copy_(0, flat_index[expert_mask], expert_output)

        combined = expert_outputs.index_select(0, flat_index).reshape(G, S, self.top_k, C)
        combined = combined * combine_weights.unsqueeze(-1)
        x = combined.sum(dim=2).reshape(-1, C)
        if pad_tokens > 0:
            x = x[:-pad_tokens]

        x = x.reshape(B, N, C)
        if return_moe_training_output is True:
            return (x, moe_training_output)

        return x


class SigmoidTopKRouter(nn.Module):
    """
    Normalized sigmoid top-k router with auxiliary-loss-free load balancing

    Paper "DeepSeek-V3 Technical Report" - https://arxiv.org/abs/2412.19437.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 2, bias_update_speed: float = 0.001) -> None:
        super().__init__()
        assert top_k >= 2, "top_k must be at least 2"

        self.top_k = top_k
        self.bias_update_speed = bias_update_speed
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.expert_bias = nn.Buffer(torch.zeros(num_experts))

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_expert_bias(self, expert_load: torch.Tensor) -> None:
        expert_load = expert_load.to(self.expert_bias)
        update_direction = torch.sign(expert_load.mean() - expert_load)
        self.expert_bias.add_(update_direction, alpha=self.bias_update_speed)

        # Expert selection is invariant to a shared bias offset.
        # Remove it so non-zero-sum sign updates cannot accumulate an unbounded common mode.
        self.expert_bias.sub_(self.expert_bias.mean())

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        affinity_scores = logits.float().sigmoid()
        expert_indices = (affinity_scores + self.expert_bias.float()).topk(self.top_k, dim=-1).indices
        expert_weights = affinity_scores.gather(-1, expert_indices)
        normalizer = expert_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(expert_weights.dtype).tiny)
        expert_weights = expert_weights / normalizer
        if token_mask is not None:
            expert_weights = expert_weights * token_mask.unsqueeze(-1)

        return (expert_indices, expert_weights)


class ExpertChoiceRouter(nn.Module):
    """
    Expert-choice router with a fixed token capacity per expert

    Paper "Mixture-of-Experts with Expert Choice Routing" - https://arxiv.org/abs/2202.09368.
    """

    def __init__(self, dim: int, num_experts: int, capacity_factor: float = 2.0) -> None:
        super().__init__()
        assert 0.0 < capacity_factor <= num_experts, "capacity_factor must be in (0, num_experts]"

        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def _capacity(self, group_size: int) -> int:
        return math.ceil(group_size * self.capacity_factor / self.num_experts)

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if token_mask is not None:
            token_mask = token_mask.to(dtype=torch.bool)
            x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)

        logits = self.gate(x)
        affinity_scores = logits.float().softmax(dim=-1)
        expert_scores = affinity_scores.transpose(-2, -1)

        if token_mask is not None:
            selection_scores = expert_scores.masked_fill(~token_mask.unsqueeze(-2), -torch.inf)
        else:
            selection_scores = expert_scores

        # Unlike the original paper, which forms routing groups across the batch, we route each sample independently.
        # This keeps inference routing invariant to batch size and ordering.
        capacity = self._capacity(x.size(-2))
        token_indices = selection_scores.topk(capacity, dim=-1).indices
        expert_weights = expert_scores.gather(-1, token_indices)
        if token_mask is not None:
            selected_token_mask = token_mask.unsqueeze(-2).expand_as(expert_scores).gather(-1, token_indices)
            expert_weights = expert_weights.masked_fill(~selected_token_mask, 0.0)

        return (token_indices, expert_weights)


class GroupedLinear(nn.Module):
    """
    Expert-major linear projection for dense and packed expert batches
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        if bias is True:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter("bias", None)

        # Weight initialization
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, *, offsets: Optional[torch.Tensor] = None, expert_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if offsets is not None:
            output = F.grouped_mm(x, self.weight.to(dtype=x.dtype).transpose(-2, -1), offs=offsets)
            if self.bias is not None:
                # Gather in FP32 so backward accumulates repeated expert-bias contributions in FP32
                output = output + self.bias.float().index_select(0, expert_indices).to(dtype=output.dtype)

        else:
            output = torch.bmm(x, self.weight.transpose(-2, -1))
            if self.bias is not None:
                bias = self.bias.unsqueeze(1)
                if output.dtype in (torch.float16, torch.bfloat16):
                    # Cast after expansion so the broadcast reduction accumulates bias gradients in FP32
                    bias = bias.float().expand_as(output)

                output = output + bias.to(dtype=output.dtype)

        return output


class GroupedSwiGLU_FFN(nn.Module):
    """
    Expert-major SwiGLU feed-forward layer for dense and packed expert batches
    """

    def __init__(
        self, in_features: int, hidden_features: int, num_experts: int, bias: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_experts = num_experts
        self.fc1_g = GroupedLinear(in_features, hidden_features, num_experts, bias)
        self.fc1_x = GroupedLinear(in_features, hidden_features, num_experts, bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = GroupedLinear(hidden_features, in_features, num_experts, bias)
        self.drop2 = nn.Dropout(dropout)

        if self.fc1_g.bias is not None:
            nn.init.normal_(self.fc1_g.weight, std=1e-6)
            nn.init.ones_(self.fc1_g.bias)

    def forward(self, x: torch.Tensor, *, expert_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process dense expert batches or packed contiguous expert segments
        """

        empty_packed = False
        offsets = None
        expert_indices = None
        if expert_counts is not None:
            if x.size(0) == 0:
                empty_packed = True
                x = x.reshape(self.num_experts, 0, self.in_features)
            else:
                # grouped_mm does not participate in autocast itself
                x = x.to(dtype=torch.bfloat16)
                offsets = expert_counts.cumsum(0, dtype=torch.int32)
                if self.fc1_g.bias is not None:
                    expert_indices = torch.arange(self.num_experts, device=x.device).repeat_interleave(
                        expert_counts, output_size=x.size(0)
                    )

        x_gate = self.fc1_g(x, offsets=offsets, expert_indices=expert_indices)
        x = self.fc1_x(x, offsets=offsets, expert_indices=expert_indices)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.fc2(x, offsets=offsets, expert_indices=expert_indices)
        x = self.drop2(x)
        if empty_packed is True:
            x = x.reshape(0, self.in_features)

        return x


def group_experts(experts: nn.ModuleList) -> GroupedSwiGLU_FFN:
    """
    Copy homogeneous SwiGLU experts configured by MoE_FFN into grouped parameters
    """

    if len(experts) == 0:
        raise ValueError("experts must contain at least one SwiGLU_FFN")

    reference = experts[0]
    for expert in experts:
        if any(module.training != experts.training for module in (expert, expert.drop1, expert.drop2)):
            raise ValueError("all experts and dropout layers must have the same training mode as the ModuleList")
        if any(
            parameter.requires_grad != reference_parameter.requires_grad
            for parameter, reference_parameter in zip(expert.parameters(), reference.parameters())
        ):
            raise ValueError("corresponding expert parameters must use the same requires_grad setting")

    with torch.device("meta"):
        grouped = GroupedSwiGLU_FFN(
            reference.fc1_g.in_features,
            reference.fc1_g.out_features,
            len(experts),
            bias=reference.fc1_g.bias is not None,
            dropout=reference.drop1.p,
        )
    with torch.no_grad():
        for projection_name in ("fc1_g", "fc1_x", "fc2"):
            grouped_projection = getattr(grouped, projection_name)
            expert_projections = [getattr(expert, projection_name) for expert in experts]
            grouped_projection.weight = nn.Parameter(
                torch.stack([projection.weight for projection in expert_projections]),
                requires_grad=expert_projections[0].weight.requires_grad,
            )
            if grouped_projection.bias is not None:
                grouped_projection.bias = nn.Parameter(
                    torch.stack([projection.bias for projection in expert_projections]),
                    requires_grad=expert_projections[0].bias.requires_grad,
                )

    return grouped.train(experts.training)  # type: ignore[no-any-return]


def ungroup_experts(grouped: GroupedSwiGLU_FFN) -> nn.ModuleList:
    """
    Copy grouped experts configured by MoE_FFN into a ModuleList of SwiGLU FFNs
    """

    if grouped.drop1.training != grouped.training or grouped.drop2.training != grouped.training:
        raise ValueError("dropout layers must have the same training mode as the grouped experts")

    num_experts = grouped.num_experts
    hidden_features = grouped.hidden_features
    in_features = grouped.in_features
    bias = grouped.fc1_g.bias is not None
    with torch.device("meta"):
        experts = nn.ModuleList(
            [SwiGLU_FFN(in_features, hidden_features, bias=bias, dropout=grouped.drop1.p) for _ in range(num_experts)]
        )
    with torch.no_grad():
        for expert_idx, expert in enumerate(experts):
            for projection_name in ("fc1_g", "fc1_x", "fc2"):
                grouped_projection = getattr(grouped, projection_name)
                projection = getattr(expert, projection_name)
                projection.weight = nn.Parameter(
                    grouped_projection.weight[expert_idx].clone(),
                    requires_grad=grouped_projection.weight.requires_grad,
                )
                if projection.bias is not None:
                    projection.bias = nn.Parameter(
                        grouped_projection.bias[expert_idx].clone(),
                        requires_grad=grouped_projection.bias.requires_grad,
                    )

    return experts.train(grouped.training)


class MoE_FFN(BaseSparseMoE_FFN):
    """
    Fine-grained MoE feed-forward layer with routed and shared experts and configurable routing

    The shared-expert and fine-grained routed-expert design follows DeepSeekMoE - https://arxiv.org/abs/2401.06066.
    Token-choice routing follows DeepSeek-V3's normalized sigmoid routing and correction-bias load balancing. The
    routed-expert output supports configurable scaling, while node-limited routing and the complementary sequence-wise
    balance loss are not implemented.
    Expert-choice routing is provided as an independent alternative.

    Setting 'grouped_token_choice=True' stores token-choice routed experts in grouped parameters and executes them with
    grouped matrix multiplications. This is intended for compatible CUDA BF16 execution, such as BF16 AMP or FSDP
    mixed-precision training, the default individual expert modules remain the portable option for other environments.

    When special-token experts are configured, the prefix identified by 'num_special_tokens' bypasses the router and
    is processed by every special-token expert. Shared experts continue to process the full sequence.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        dropout: float = 0.0,
        num_routed_experts: int = 32,
        num_shared_experts: int = 1,
        num_special_token_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        routing_type: Literal["token_choice", "expert_choice"] = "token_choice",
        grouped_token_choice: bool = False,
        top_k: int = 2,
        router_bias_update_speed: float = 0.001,
        expert_choice_capacity_factor: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_routed_experts = num_routed_experts
        self.has_special_token_experts = num_special_token_experts > 0
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_type = routing_type
        if routing_type == "token_choice":
            self.router = SigmoidTopKRouter(
                in_features,
                num_routed_experts,
                top_k=top_k,
                bias_update_speed=router_bias_update_speed,
            )
        elif routing_type == "expert_choice":
            self.router = ExpertChoiceRouter(
                in_features,
                num_routed_experts,
                capacity_factor=expert_choice_capacity_factor,
            )
        else:
            raise ValueError(f"Unknown routing_type '{routing_type}'")

        self.shared_experts = nn.ModuleList(
            [SwiGLU_FFN(in_features, hidden_features, bias=bias, dropout=dropout) for _ in range(num_shared_experts)]
        )
        self.special_token_experts = nn.ModuleList(
            [
                SwiGLU_FFN(in_features, hidden_features, bias=bias, dropout=dropout)
                for _ in range(num_special_token_experts)
            ]
        )
        self.routed_experts: nn.ModuleList | GroupedSwiGLU_FFN
        if routing_type == "expert_choice" or grouped_token_choice is True:
            self.routed_experts = GroupedSwiGLU_FFN(in_features, hidden_features, num_routed_experts, bias, dropout)
        else:
            self.routed_experts = nn.ModuleList(
                [
                    SwiGLU_FFN(in_features, hidden_features, bias=bias, dropout=dropout)
                    for _ in range(num_routed_experts)
                ]
            )

    def _route_token_choice(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = x.numel() // self.in_features
        expert_indices, expert_weights = self.router(x, token_mask=token_mask)
        num_choices = expert_indices.size(-1)
        flat_expert_indices = expert_indices.reshape(-1)
        flat_expert_weights = expert_weights.reshape(-1)
        flat_token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(-1).expand(-1, num_choices).reshape(-1)
        if token_mask is not None:
            flat_assignment_mask = (
                token_mask.to(dtype=torch.bool).reshape(num_tokens, 1).expand(-1, num_choices).reshape(-1)
            )
        else:
            flat_assignment_mask = None

        return (flat_token_indices, flat_expert_indices, flat_expert_weights, flat_assignment_mask)

    def _route_expert_choice(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_indices, expert_weights = self.router(x, token_mask=token_mask)
        group_size = x.size(-2)
        num_groups = math.prod(x.shape[:-2])
        capacity = token_indices.size(-1)
        grouped_token_indices = token_indices.reshape(num_groups, self.num_routed_experts, capacity)
        group_offsets = torch.arange(num_groups, device=x.device).reshape(-1, 1, 1) * group_size

        token_indices = (grouped_token_indices + group_offsets).transpose(0, 1)
        token_indices = token_indices.reshape(self.num_routed_experts, num_groups * capacity)
        expert_weights = expert_weights.reshape(num_groups, self.num_routed_experts, capacity).transpose(0, 1)
        expert_weights = expert_weights.reshape(self.num_routed_experts, num_groups * capacity)

        return (token_indices, expert_weights)

    def _run_grouped_experts(
        self,
        flat_routed_x: torch.Tensor,
        token_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_counts: Optional[torch.Tensor],
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        flat_token_indices = token_indices.reshape(-1)

        # The gather's backward scatter-add must accumulate repeated token gradients in FP32
        gather_source = flat_routed_x
        if gather_source.dtype in (torch.float16, torch.bfloat16):
            gather_source = gather_source.float()

        expert_input = gather_source.index_select(0, flat_token_indices).to(dtype=flat_routed_x.dtype)
        if token_mask is not None:
            selected_mask = token_mask.reshape(-1).to(dtype=torch.bool).index_select(0, flat_token_indices)
            expert_input = expert_input.masked_fill(~selected_mask.unsqueeze(-1), 0.0)

        expert_input = expert_input.reshape(*token_indices.shape, self.in_features)
        expert_output = self.routed_experts(expert_input, expert_counts=expert_counts).reshape(-1, self.in_features)

        # Accumulate the forward scatter-add's low-precision expert contributions in FP32
        combine_dtype = torch.float32 if expert_output.dtype in (torch.float16, torch.bfloat16) else expert_output.dtype
        combine_weights = expert_weights.reshape(-1, 1).to(dtype=combine_dtype)
        routed_output = expert_output.new_zeros((flat_routed_x.size(0), self.in_features), dtype=combine_dtype)
        routed_output.index_add_(0, flat_token_indices, expert_output.to(dtype=combine_dtype) * combine_weights)

        return routed_output.to(dtype=expert_output.dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        num_special_tokens: int = 0,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        routed_token_mask: Optional[torch.Tensor]
        if self.has_special_token_experts is True:
            routed_x = x[:, num_special_tokens:]
            if token_mask is not None:
                routed_token_mask = token_mask[:, num_special_tokens:]
            else:
                routed_token_mask = None
        else:
            routed_x = x
            routed_token_mask = token_mask

        num_routed_tokens = routed_x.numel() // self.in_features
        flat_routed_x = routed_x.reshape(num_routed_tokens, self.in_features)
        expert_loads: Optional[torch.Tensor] = None
        if isinstance(self.routed_experts, nn.ModuleList):
            flat_token_indices, flat_expert_indices, flat_expert_weights, flat_assignment_mask = (
                self._route_token_choice(routed_x, routed_token_mask)
            )
            if torch.is_autocast_enabled(x.device.type) is True:
                expert_dtype = torch.get_autocast_dtype(x.device.type)
            else:
                expert_dtype = x.dtype

            routed_output = x.new_zeros((num_routed_tokens, self.in_features), dtype=expert_dtype)
            for expert_idx, expert in enumerate(self.routed_experts):
                expert_mask = flat_expert_indices == expert_idx
                if flat_assignment_mask is not None:
                    expert_mask = expert_mask & flat_assignment_mask

                token_indices = flat_token_indices[expert_mask]
                expert_input = flat_routed_x.index_select(0, token_indices)
                expert_output = expert(expert_input)
                combine_weights = flat_expert_weights[expert_mask].to(dtype=expert_output.dtype).unsqueeze(-1)
                routed_output.index_add_(0, token_indices, expert_output * combine_weights)

            if self.training is True and return_moe_training_output is True:
                selected_experts = flat_expert_indices
                if flat_assignment_mask is not None:
                    selected_experts = selected_experts[flat_assignment_mask]

                expert_loads = torch.bincount(
                    selected_experts.reshape(-1), minlength=self.num_routed_experts
                ).unsqueeze(0)

        else:
            expert_counts = None
            selected_token_mask = routed_token_mask
            if self.routing_type == "token_choice":
                flat_token_indices, flat_expert_indices, flat_expert_weights, flat_assignment_mask = (
                    self._route_token_choice(routed_x, routed_token_mask)
                )
                if flat_assignment_mask is not None:
                    flat_expert_indices = flat_expert_indices[flat_assignment_mask]
                    flat_token_indices = flat_token_indices[flat_assignment_mask]
                    flat_expert_weights = flat_expert_weights[flat_assignment_mask]

                order = flat_expert_indices.argsort(stable=True)
                expert_counts = flat_expert_indices.new_zeros(self.num_routed_experts)
                expert_counts.scatter_add_(0, flat_expert_indices, torch.ones_like(flat_expert_indices))
                token_indices = flat_token_indices[order]
                expert_weights = flat_expert_weights[order]
                selected_token_mask = None
                if self.training is True and return_moe_training_output is True:
                    expert_loads = expert_counts.unsqueeze(0)

            else:
                token_indices, expert_weights = self._route_expert_choice(routed_x, routed_token_mask)

            routed_output = self._run_grouped_experts(
                flat_routed_x,
                token_indices,
                expert_weights,
                expert_counts,
                selected_token_mask,
            )
            expert_dtype = routed_output.dtype

        routed_output = (routed_output * self.routed_scaling_factor).reshape_as(routed_x)

        # Special token experts (if any)
        if self.has_special_token_experts is True:
            special_output = x.new_zeros(
                (x.size(0), num_special_tokens, self.in_features),
                dtype=expert_dtype,
            )
            special_x = x[:, :num_special_tokens]
            for expert in self.special_token_experts:
                special_output = special_output + expert(special_x)

            output = torch.concat((special_output, routed_output), dim=1)
        else:
            output = routed_output

        # Shared experts (if any)
        for expert in self.shared_experts:
            output = output + expert(x)

        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1)

        if return_moe_training_output is True:
            moe_training_output = _empty_moe_training_output(output)
            if expert_loads is not None:
                moe_training_output["expert_loads"] = expert_loads

            return (output, moe_training_output)

        return output

    def update_expert_bias(self, expert_load: torch.Tensor) -> None:
        self.router.update_expert_bias(expert_load)
