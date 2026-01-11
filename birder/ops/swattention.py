import torch
import torch.nn.functional as F
from torch import nn
from torch.library import custom_op

from birder.kernels.load_kernel import load_swattention

SWATTENTION_CUDA_NUM_THREADS = 128
SWATTENTION = None


def set_swattention_num_threads(num_threads: int) -> None:
    global SWATTENTION_CUDA_NUM_THREADS  # pylint: disable=global-statement
    SWATTENTION_CUDA_NUM_THREADS = num_threads


# SWAttention QK RPB
####################


@custom_op("birder::swattention_qk_rpb", mutates_args=())  # type: ignore[untyped-decorator]
def swattention_qk_rpb_op(
    query: torch.Tensor, key: torch.Tensor, rpb: torch.Tensor, height: int, width: int, kernel_size: int
) -> torch.Tensor:
    return SWATTENTION.qk_rpb_forward(  # type: ignore[attr-defined]
        query, key, rpb, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
    )


@swattention_qk_rpb_op.register_fake  # type: ignore[untyped-decorator]
def _swattention_qk_rpb_fake(  # pylint: disable=unused-argument
    query: torch.Tensor, key: torch.Tensor, rpb: torch.Tensor, height: int, width: int, kernel_size: int
) -> torch.Tensor:
    local_len = kernel_size * kernel_size
    return query.new_empty((*query.shape[:-1], local_len))


def _swattention_qk_rpb_setup_context(  # type: ignore[no-untyped-def] # pylint: disable=unused-argument
    ctx, inputs, output
) -> None:
    (query, key, _rpb, height, width, kernel_size) = inputs
    ctx.save_for_backward(query, key)
    ctx.height = height
    ctx.width = width
    ctx.kernel_size = kernel_size


def _swattention_qk_rpb_backward(ctx, grad_output):  # type: ignore[no-untyped-def]
    (query, key) = ctx.saved_tensors
    (d_query, d_key, d_rpb) = swattention_qk_rpb_backward_op(
        grad_output.contiguous(), query, key, ctx.height, ctx.width, ctx.kernel_size
    )
    return (d_query, d_key, d_rpb, None, None, None)


swattention_qk_rpb_op.register_autograd(_swattention_qk_rpb_backward, setup_context=_swattention_qk_rpb_setup_context)


@custom_op("birder::swattention_qk_rpb_backward", mutates_args=())  # type: ignore[untyped-decorator]
def swattention_qk_rpb_backward_op(
    d_attn_weight: torch.Tensor, query: torch.Tensor, key: torch.Tensor, height: int, width: int, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return SWATTENTION.qk_rpb_backward(  # type: ignore
        d_attn_weight, query, key, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
    )


@swattention_qk_rpb_backward_op.register_fake  # type: ignore[untyped-decorator]
def _swattention_qk_rpb_backward_fake(  # pylint: disable=unused-argument
    d_attn_weight: torch.Tensor, query: torch.Tensor, key: torch.Tensor, height: int, width: int, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_len = kernel_size * kernel_size
    d_query = query.new_empty(query.shape)
    d_key = key.new_empty(key.shape)
    d_rpb = query.new_empty((query.shape[1], local_len))
    return (d_query, d_key, d_rpb)


# SWAttention AV
####################


@custom_op("birder::swattention_av", mutates_args=())  # type: ignore[untyped-decorator]
def swattention_av_op(
    attn_weight: torch.Tensor, value: torch.Tensor, height: int, width: int, kernel_size: int
) -> torch.Tensor:
    return SWATTENTION.av_forward(  # type: ignore[attr-defined]
        attn_weight, value, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
    )


@swattention_av_op.register_fake  # type: ignore[untyped-decorator]
def _swattention_av_fake(  # pylint: disable=unused-argument
    attn_weight: torch.Tensor, value: torch.Tensor, height: int, width: int, kernel_size: int
) -> torch.Tensor:
    return value.new_empty(value.shape)


def _swattention_av_setup_context(  # type: ignore[no-untyped-def] # pylint: disable=unused-argument
    ctx, inputs, output
) -> None:
    attn_weight, value, height, width, kernel_size = inputs
    ctx.save_for_backward(attn_weight, value)
    ctx.height = height
    ctx.width = width
    ctx.kernel_size = kernel_size


def _swattention_av_backward(ctx, grad_output):  # type: ignore[no-untyped-def]
    (attn_weight, value) = ctx.saved_tensors
    (d_attn_weight, d_value) = swattention_av_backward_op(
        grad_output.contiguous(), attn_weight, value, ctx.height, ctx.width, ctx.kernel_size
    )
    return (d_attn_weight, d_value, None, None, None)


swattention_av_op.register_autograd(_swattention_av_backward, setup_context=_swattention_av_setup_context)


@custom_op("birder::swattention_av_backward", mutates_args=())  # type: ignore[untyped-decorator]
def swattention_av_backward_op(
    d_output: torch.Tensor, attn_weight: torch.Tensor, value: torch.Tensor, height: int, width: int, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return SWATTENTION.av_backward(  # type: ignore
        d_output, attn_weight, value, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
    )


@swattention_av_backward_op.register_fake  # type: ignore[untyped-decorator]
def _swattention_av_backward_fake(  # pylint: disable=unused-argument
    d_output: torch.Tensor, attn_weight: torch.Tensor, value: torch.Tensor, height: int, width: int, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    d_attn_weight = attn_weight.new_empty(attn_weight.shape)
    d_value = value.new_empty(value.shape)
    return (d_attn_weight, d_value)


# pylint: disable=invalid-name
class SWAttention_QK_RPB(nn.Module):
    """
    TransNeXt: Robust Foveal Visual Perception for Vision Transformers: https://arxiv.org/abs/2311.17132

    Lazy-loading SWATTENTION operator.

    The custom kernel is loaded on first instantiation, not at import time.
    Falls back to pure PyTorch implementation if kernel loading fails.
    """

    def __init__(self) -> None:
        super().__init__()

        global SWATTENTION  # pylint: disable=global-statement
        if SWATTENTION is None and not torch.jit.is_tracing() and not torch.jit.is_scripting():
            SWATTENTION = load_swattention()

        self.is_available = SWATTENTION is not None

    def forward(
        self,
        kv: torch.Tensor,
        q_norm_scaled: torch.Tensor,
        relative_pos_bias_local: torch.Tensor,
        padding_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        window_size: int,
        local_len: int,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pure PyTorch
        if self.is_available is False or kv.is_cuda is False:
            return swattention_qk_rpb(
                kv,
                q_norm_scaled,
                relative_pos_bias_local,
                padding_mask,
                num_heads,
                head_dim,
                window_size,
                local_len,
                H,
                W,
            )

        # Custom kernel
        (B, N, _) = kv.size()

        # Generate unfolded keys and values and l2-normalize them
        (k_local, v_local) = kv.reshape(B, N, 2 * num_heads, head_dim).permute(0, 2, 1, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local = swattention_qk_rpb_op(
            q_norm_scaled.contiguous(),
            F.normalize(k_local, dim=-1).contiguous(),
            relative_pos_bias_local,
            H,
            W,
            window_size,
        )

        return (attn_local, v_local)


# pylint: disable=invalid-name
class SWAttention_AV(nn.Module):
    """
    TransNeXt: Robust Foveal Visual Perception for Vision Transformers: https://arxiv.org/abs/2311.17132

    Lazy-loading SWATTENTION operator.

    The custom kernel is loaded on first instantiation, not at import time.
    Falls back to pure PyTorch implementation if kernel loading fails.
    """

    def __init__(self) -> None:
        super().__init__()

        global SWATTENTION  # pylint: disable=global-statement
        if SWATTENTION is None and not torch.jit.is_tracing() and not torch.jit.is_scripting():
            SWATTENTION = load_swattention()

        self.is_available = SWATTENTION is not None

    def forward(
        self,
        q_norm: torch.Tensor,
        attn_local: torch.Tensor,
        v_local: torch.Tensor,
        learnable_tokens: torch.Tensor,
        learnable_bias: torch.Tensor,
        window_size: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        # Pure PyTorch
        if self.is_available is False or q_norm.is_cuda is False:
            return swattention_av(q_norm, attn_local, v_local, learnable_tokens, learnable_bias)

        # Custom kernel
        attn_local = (q_norm @ learnable_tokens) + learnable_bias + attn_local
        return swattention_av_op(attn_local.type_as(v_local), v_local.contiguous(), H, W, window_size)


def swattention_qk_rpb(
    kv: torch.Tensor,
    q_norm_scaled: torch.Tensor,
    relative_pos_bias_local: torch.Tensor,
    padding_mask: torch.Tensor,
    num_heads: int,
    head_dim: int,
    window_size: int,
    local_len: int,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    (B, N, _) = kv.size()

    # Generate unfolded keys and values and l2-normalize them
    (k_local, v_local) = kv.chunk(2, dim=-1)
    k_local = F.normalize(k_local.reshape(B, N, num_heads, head_dim), dim=-1).reshape(B, N, -1)
    kv_local = torch.concat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)

    (k_local, v_local) = (
        F.unfold(kv_local, kernel_size=window_size, padding=window_size // 2, stride=1)
        .reshape(B, 2 * num_heads, head_dim, local_len, N)
        .permute(0, 1, 4, 2, 3)
        .chunk(2, dim=1)
    )

    # Compute local similarity
    attn_local = (
        (q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) + relative_pos_bias_local.unsqueeze(1)
    ).masked_fill(padding_mask, float("-inf"))

    return (attn_local, v_local)


def swattention_av(
    q_norm: torch.Tensor,
    attn_local: torch.Tensor,
    v_local: torch.Tensor,
    learnable_tokens: torch.Tensor,
    learnable_bias: torch.Tensor,
) -> torch.Tensor:
    return (
        ((q_norm @ learnable_tokens) + learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)
    ).squeeze(-2)
