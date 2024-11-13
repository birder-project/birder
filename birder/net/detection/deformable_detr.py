"""
Deformable DETR, adapted from
https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py
and
https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py

Paper "Deformable DETR: Deformable Transformers for End-to-End Object Detection",
https://arxiv.org/abs/2010.04159
"""

# Reference license: Apache-2.0 (both)

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from birder.kernels.load_kernel import load_msda

MSDA = None


def multi_scale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor | list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    (batch_size, _, num_heads, hidden_dim) = value.size()
    (_, num_queries, num_heads, num_levels, num_points, _) = sampling_locations.size()
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # (batch_size, height*width, num_heads, hidden_dim)
        # -> (batch_size, height*width, num_heads*hidden_dim)
        # -> (batch_size, num_heads*hidden_dim, height*width)
        # -> (batch_size*num_heads, hidden_dim, height, width)
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )

        # (batch_size, num_queries, num_heads, num_points, 2)
        # -> (batch_size, num_heads, num_queries, num_points, 2)
        # -> (batch_size*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)

        # (batch_size*num_heads, hidden_dim, num_queries, num_points)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )

    return output.transpose(1, 2).contiguous()


# pylint: disable=abstract-method,arguments-differ
class MSDAFunction(Function):
    @staticmethod
    def forward(  # type: ignore
        ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(  # type: ignore[attr-defined]
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):  # type: ignore
        (value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights) = (
            ctx.saved_tensors
        )
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(  # type: ignore[attr-defined]
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return (grad_value, None, None, grad_sampling_loc, grad_attn_weight, None)


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, d_model: int, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()

        global MSDA  # pylint: disable=global-statement
        MSDA = load_msda()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")

        dim_per_head = d_model // n_heads
        # Ensure dim_per_head is power of 2
        if ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0) is False:
            raise ValueError(
                "Set d_model in MultiScaleDeformableAttention to make the dimension of each attention head a power of 2"
            )

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        (N, num_queries, _) = query.size()
        (N, sequence_length, _) = input_flatten.size()
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == sequence_length

        value = self.value_proj(input_flatten)
        value = value.view(N, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        # N, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead"
            )

        # Custom kernel
        if MSDA is not None and value.is_cuda is True:
            output = MSDAFunction.apply(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

        # Pure PyTorch
        else:
            output = multi_scale_deformable_attention(
                value, input_spatial_shapes, sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src
