import torch
import torch.nn.functional as F
from torch import nn

from birder.layers.ffn import FFN


class MultiHeadAttentionPool(nn.Module):
    """
    Adapted from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py#L12
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        qkv_bias: bool,
        latent_len: int = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.mlp = FFN(dim, mlp_dim, act_layer=nn.GELU)

        # Weight initialization
        nn.init.trunc_normal_(self.latent, std=dim**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.size()

        q_latent = self.latent.expand(B, self.latent_len, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (k, v) = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # pylint: disable=not-callable
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = x + self.mlp(self.norm(x))

        return x
