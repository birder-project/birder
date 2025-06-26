import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from birder.kernels.load_kernel import load_msda

MSDA = None


class MultiScaleDeformableAttention(nn.Module):
    """
    Deformable DETR: Deformable Transformers for End-to-End Object Detection: https://arxiv.org/abs/2010.04159

    Lazy-loading MSDA operator.

    The custom kernel is loaded on first instantiation, not at import time.
    Falls back to pure PyTorch implementation if kernel loading fails.
    """

    def __init__(self) -> None:
        super().__init__()

        global MSDA  # pylint: disable=global-statement
        if MSDA is None and not torch.jit.is_tracing() and not torch.jit.is_scripting():
            MSDA = load_msda()

        self.is_available = MSDA is not None

    def forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: int,
    ) -> torch.Tensor:
        # Pure PyTorch
        if self.is_available is False or value.is_cuda is False:
            return multi_scale_deformable_attention(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
            )

        # Custom kernel
        return MSDAFunction.apply(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
        )


# pylint: disable=abstract-method,arguments-differ
class MSDAFunction(Function):
    @staticmethod
    @torch.compiler.disable()
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(  # type: ignore
        ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(  # type: ignore[union-attr]
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @torch.compiler.disable()
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):  # type: ignore
        (value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights) = (
            ctx.saved_tensors
        )
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(  # type: ignore[union-attr]
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return (grad_value, None, None, grad_sampling_loc, grad_attn_weight, None)


def multi_scale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,  # pylint:disable=unused-argument
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,  # pylint:disable=unused-argument
) -> torch.Tensor:
    (batch_size, _, num_heads, hidden_dim) = value.size()
    (_, num_queries, num_heads, num_levels, num_points, _) = sampling_locations.size()
    areas: list[int] = value_spatial_shapes.prod(dim=1).tolist()
    value_list = value.split(areas, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, spatial_shape in enumerate(value_spatial_shapes):
        # (batch_size, height*width, num_heads, hidden_dim)
        # -> (batch_size, height*width, num_heads*hidden_dim)
        # -> (batch_size, num_heads*hidden_dim, height*width)
        # -> (batch_size*num_heads, hidden_dim, height, width)
        height = spatial_shape[0]
        width = spatial_shape[1]
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
