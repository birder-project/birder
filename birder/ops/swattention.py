import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from birder.kernels.load_kernel import load_swattention

SWATTENTION_CUDA_NUM_THREADS = 128
SWATTENTION = None


def set_swattention_num_threads(num_threads: int) -> None:
    global SWATTENTION_CUDA_NUM_THREADS  # pylint: disable=global-statement
    SWATTENTION_CUDA_NUM_THREADS = num_threads


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
        attn_local = SWAttentionFunction_QK_RPB.apply(
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
        return SWAttentionFunction_AV.apply(attn_local.type_as(v_local), v_local.contiguous(), H, W, window_size)


# pylint: disable=abstract-method,arguments-differ,invalid-name
class SWAttentionFunction_QK_RPB(Function):
    @staticmethod
    def forward(ctx, query, key, rpb, height, width, kernel_size):  # type: ignore
        attn_weight = SWATTENTION.qk_rpb_forward(  # type: ignore[union-attr]
            query, key, rpb, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
        )

        ctx.save_for_backward(query, key)
        ctx.height = height
        ctx.width = width
        ctx.kernel_size = kernel_size

        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):  # type: ignore
        (query, key) = ctx.saved_tensors
        height = ctx.height
        width = ctx.width
        kernel_size = ctx.kernel_size

        (d_query, d_key, d_rpb) = SWATTENTION.qk_rpb_backward(  # type: ignore[union-attr]
            d_attn_weight.contiguous(), query, key, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
        )

        return (d_query, d_key, d_rpb, None, None, None)


# pylint: disable=abstract-method,arguments-differ,invalid-name
class SWAttentionFunction_AV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, kernel_size):  # type: ignore
        output = SWATTENTION.av_forward(  # type: ignore[union-attr]
            attn_weight, value, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
        )

        ctx.save_for_backward(attn_weight, value)
        ctx.height = height
        ctx.width = width
        ctx.kernel_size = kernel_size

        return output

    @staticmethod
    def backward(ctx, d_output):  # type: ignore
        (attn_weight, value) = ctx.saved_tensors
        height = ctx.height
        width = ctx.width
        kernel_size = ctx.kernel_size

        d_attn_weight, d_value = SWATTENTION.av_backward(  # type: ignore[union-attr]
            d_output.contiguous(), attn_weight, value, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
        )

        return (d_attn_weight, d_value, None, None, None)


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
