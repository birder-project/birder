import torch
import torch.nn.functional as F
from torch import nn
from torch.library import custom_op

from birder.kernels.load_kernel import load_msda

MSDA = None


@custom_op("birder::ms_deform_attn", mutates_args=())  # type: ignore[untyped-decorator]
def ms_deform_attn_op(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    return MSDA.ms_deform_attn_forward(  # type: ignore[attr-defined]
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    )


@custom_op("birder::ms_deform_attn_backward", mutates_args=())  # type: ignore[untyped-decorator]
def ms_deform_attn_backward_op(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return MSDA.ms_deform_attn_backward(  # type: ignore
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        grad_output,
        im2col_step,
    )


@custom_op("birder::ms_deform_attn_packed", mutates_args=())  # type: ignore[untyped-decorator]
def ms_deform_attn_packed_op(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_per_level: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    return MSDA.ms_deform_attn_packed_forward(  # type: ignore[attr-defined]
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        num_points_per_level,
        im2col_step,
    )


@custom_op("birder::ms_deform_attn_packed_backward", mutates_args=())  # type: ignore[untyped-decorator]
def ms_deform_attn_packed_backward_op(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_per_level: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return MSDA.ms_deform_attn_packed_backward(  # type: ignore[attr-defined,no-any-return]
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        num_points_per_level,
        grad_output,
        im2col_step,
    )


@ms_deform_attn_op.register_fake  # type: ignore[untyped-decorator]
def _ms_deform_attn_fake(  # pylint: disable=unused-argument
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    batch_size = value.shape[0]
    num_queries = sampling_locations.shape[1]
    num_heads = value.shape[2]
    hidden_dim = value.shape[3]
    return value.new_empty((batch_size, num_queries, num_heads * hidden_dim))


@ms_deform_attn_backward_op.register_fake  # type: ignore[untyped-decorator]
def _ms_deform_attn_backward_fake(  # pylint: disable=unused-argument
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_value = value.new_empty(value.shape)
    grad_sampling_locations = sampling_locations.new_empty(sampling_locations.shape)
    grad_attention_weights = attention_weights.new_empty(attention_weights.shape)
    return (grad_value, grad_sampling_locations, grad_attention_weights)


@ms_deform_attn_packed_op.register_fake  # type: ignore[untyped-decorator]
def _ms_deform_attn_packed_fake(  # pylint: disable=unused-argument
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_per_level: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    batch_size = value.shape[0]
    num_queries = sampling_locations.shape[1]
    num_heads = value.shape[2]
    hidden_dim = value.shape[3]
    return value.new_empty((batch_size, num_queries, num_heads * hidden_dim))


@ms_deform_attn_packed_backward_op.register_fake  # type: ignore[untyped-decorator]
def _ms_deform_attn_packed_backward_fake(  # pylint: disable=unused-argument
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_per_level: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_value = value.new_empty(value.shape)
    grad_sampling_locations = sampling_locations.new_empty(sampling_locations.shape)
    grad_attention_weights = attention_weights.new_empty(attention_weights.shape)
    return (grad_value, grad_sampling_locations, grad_attention_weights)


def _ms_deform_attn_setup_context(  # type: ignore[no-untyped-def] # pylint: disable=unused-argument
    ctx, inputs, output
) -> None:
    (
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ) = inputs
    ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
    ctx.im2col_step = im2col_step


def _ms_deform_attn_backward(ctx, grad_output):  # type: ignore[no-untyped-def]
    value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
    grad_value, grad_sampling_loc, grad_attn_weight = ms_deform_attn_backward_op(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        grad_output,
        ctx.im2col_step,
    )
    return (grad_value, None, None, grad_sampling_loc, grad_attn_weight, None)


ms_deform_attn_op.register_autograd(_ms_deform_attn_backward, setup_context=_ms_deform_attn_setup_context)


def _ms_deform_attn_packed_setup_context(  # type: ignore[no-untyped-def] # pylint: disable=unused-argument
    ctx, inputs, output
) -> None:
    (
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        num_points_per_level,
        im2col_step,
    ) = inputs
    ctx.save_for_backward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        num_points_per_level,
    )
    ctx.im2col_step = im2col_step


def _ms_deform_attn_packed_backward(ctx, grad_output):  # type: ignore[no-untyped-def]
    (
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        num_points_per_level,
    ) = ctx.saved_tensors
    grad_value, grad_sampling_loc, grad_attn_weight = ms_deform_attn_packed_backward_op(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        num_points_per_level,
        grad_output.contiguous(),
        ctx.im2col_step,
    )
    return (grad_value, None, None, grad_sampling_loc, grad_attn_weight, None, None)


ms_deform_attn_packed_op.register_autograd(
    _ms_deform_attn_packed_backward, setup_context=_ms_deform_attn_packed_setup_context
)


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
        src_shapes: list[list[int]],
    ) -> torch.Tensor:
        # Pure PyTorch path
        if self.is_available is False or value.is_cuda is False:
            return multi_scale_deformable_attention(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
                src_shapes,
            )

        # Match grid_sample's autocast policy
        # The CUDA kernel requires its floating-point inputs to have the same dtype
        # if torch.is_autocast_enabled(value.device.type) is True:
        if torch.is_autocast_enabled() is True:  # TorchScript compatibility
            kernel_dtype = torch.promote_types(value.dtype, sampling_locations.dtype)
            kernel_dtype = torch.promote_types(kernel_dtype, attention_weights.dtype)
            value = value.to(dtype=kernel_dtype)
            sampling_locations = sampling_locations.to(dtype=kernel_dtype)
            attention_weights = attention_weights.to(dtype=kernel_dtype)

        return ms_deform_attn_op(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
        )

    def forward_packed(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        num_points_per_level: torch.Tensor,
        im2col_step: int,
        src_shapes: list[list[int]],
        num_points_per_level_list: list[int],
    ) -> torch.Tensor:
        if self.is_available is False or value.is_cuda is False:
            return multi_scale_deformable_attention_packed(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                num_points_per_level_list,
                im2col_step,
                src_shapes,
            )

        # Match grid_sample's autocast policy
        # The CUDA kernel requires its floating-point inputs to have the same dtype
        # if torch.is_autocast_enabled(value.device.type) is True:
        if torch.is_autocast_enabled() is True:  # TorchScript compatibility
            kernel_dtype = torch.promote_types(value.dtype, sampling_locations.dtype)
            kernel_dtype = torch.promote_types(kernel_dtype, attention_weights.dtype)
            value = value.to(dtype=kernel_dtype)
            sampling_locations = sampling_locations.to(dtype=kernel_dtype)
            attention_weights = attention_weights.to(dtype=kernel_dtype)

        return ms_deform_attn_packed_op(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            num_points_per_level,
            im2col_step,
        )


def multi_scale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,  # pylint: disable=unused-argument
    value_level_start_index: torch.Tensor,  # pylint: disable=unused-argument
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,  # pylint: disable=unused-argument
    src_shapes: list[list[int]],
) -> torch.Tensor:
    """
    Pure PyTorch implementation of multi-scale deformable attention
    """

    batch_size, _, num_heads, hidden_dim = value.size()
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.size()

    # assert len(src_shapes) == num_levels
    level_sizes = [H * W for H, W in src_shapes]
    sampling_grids = 2 * sampling_locations - 1

    # (B, Len_in, heads, head_dim) -> (B, heads, head_dim, Len_in)
    value = value.permute(0, 2, 3, 1).contiguous()
    value_list = value.split(level_sizes, dim=-1)

    # (B, Len_q, heads, levels, points) -> (B*heads, 1, Len_q, levels, points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels, num_points
    )

    output = value.new_zeros((batch_size * num_heads, hidden_dim, num_queries))
    for level_id, (H, W) in enumerate(src_shapes):
        value_l_ = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, H, W)
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level_id]
            .transpose(1, 2)
            .reshape(batch_size * num_heads, num_queries, num_points, 2)
        )

        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        output += (sampling_value_l_ * attention_weights[:, :, :, level_id, :]).sum(-1)

    output = output.view(batch_size, num_heads * hidden_dim, num_queries)

    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attention_packed(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,  # pylint: disable=unused-argument
    value_level_start_index: torch.Tensor,  # pylint: disable=unused-argument
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_per_level: list[int],
    im2col_step: int,  # pylint: disable=unused-argument
    src_shapes: list[list[int]],
) -> torch.Tensor:
    """
    Pure PyTorch implementation of packed multi-scale deformable attention
    """

    batch_size, _, num_heads, hidden_dim = value.size()
    num_queries = sampling_locations.size(1)
    level_sizes = [H * W for H, W in src_shapes]

    value = value.permute(0, 2, 3, 1).flatten(0, 1)
    value_list = value.split(level_sizes, dim=-1)
    sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4)
    sampling_locations_list = sampling_locations.split(num_points_per_level, dim=-2)

    sampling_value_list = []
    for level, (H, W) in enumerate(src_shapes):
        value_level = value_list[level].reshape(batch_size * num_heads, hidden_dim, H, W)
        sampling_grid_level = (2 * sampling_locations_list[level] - 1).flatten(0, 1)
        sampling_value_level = F.grid_sample(
            value_level,
            sampling_grid_level,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_level)

    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_per_level)
    )
    output = (torch.concat(sampling_value_list, dim=-1) * attention_weights).sum(-1)
    output = output.reshape(batch_size, num_heads * hidden_dim, num_queries)

    return output.transpose(1, 2).contiguous()
