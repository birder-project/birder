"""
CAPI, adapted from
https://github.com/facebookresearch/capi/blob/main/model.py

Paper "Cluster and Predict Latent Patches for Improved Masked Image Modeling",
https://arxiv.org/abs/2502.08769

Changes from original:
* Replaced decoder RoPE with simple sin-cos embedding
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP

from birder.common import masking
from birder.common import training_utils
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.ssl.base import SSLBaseNet

exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}


def stable_exp(M: torch.Tensor) -> torch.Tensor:
    shift = M.max(dim=-2, keepdim=True).values
    if training_utils.is_dist_available_and_initialized() is True:
        torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)

    M += exp_max_values[M.dtype] - shift

    return M.exp()


def reduced_sum(*args: Any, **kwargs: Any) -> torch.Tensor:
    summed = torch.sum(*args, **kwargs)
    if training_utils.is_dist_available_and_initialized() is True:
        torch.distributed.all_reduce(summed)

    return summed


@torch.no_grad()  # type: ignore[misc]
def sinkhorn_knopp(M: torch.Tensor, n_iterations: int, eps: float = 1e-8) -> torch.Tensor:
    M = stable_exp(M)
    for _ in range(n_iterations):
        M /= reduced_sum(M, dim=-2, keepdim=True) + eps
        M /= torch.sum(M, dim=-1, keepdim=True) + eps

    return M


class OnlineClustering(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool,
        n_sk_iter: int,
        target_temp: float,
        pred_temp: float,
        positionwise_sk: bool = True,
    ):
        super().__init__()
        self.n_sk_iter = n_sk_iter
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        self.positionwise_sk = positionwise_sk
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)

        # Weight initialization
        nn.init.normal_(self.layer.weight, std=1.0)
        if bias is True:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_n = F.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n)
        if self.positionwise_sk is False:
            logits = logits.flatten(0, -2)

        assignments = sinkhorn_knopp(logits.detach() / self.target_temp, n_iterations=self.n_sk_iter)
        tgt = assignments.flatten(0, -2).float()
        pred = logits.flatten(0, -2).float()
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()

        return (assignments.detach(), loss)


class L2NormLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, do_weight_norm: bool = True):
        super().__init__()
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if do_weight_norm is True:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


class CrossAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(decoder_dim, decoder_dim)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2)
        self.proj = nn.Linear(decoder_dim, decoder_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        (B, N, C) = tgt.size()
        n_kv = memory.size(1)
        q = self.q(tgt).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(B, n_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (k, v) = kv.unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # pylint: disable=not-callable
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(decoder_dim, eps=1e-5)
        self.cross_attn = CrossAttention(encoder_dim, decoder_dim, num_heads=num_heads)
        self.norm2 = nn.RMSNorm(decoder_dim, eps=1e-5)
        self.mlp = MLP(decoder_dim, [int(decoder_dim * mlp_ratio), decoder_dim], activation_layer=nn.GELU)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = tgt + self.cross_attn(self.norm1(tgt), memory)
        x = x + self.mlp(self.norm2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_size: tuple[int, int], embed_dim: int, depth: int) -> None:
        super().__init__()

        encoder_dim = embed_dim
        decoder_embed_dim = encoder_dim
        decoder_depth = depth
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sin-cos embedding
        pos_embedding = pos_embedding_sin_cos_2d(
            h=input_size[0],
            w=input_size[1],
            dim=decoder_embed_dim,
            num_special_tokens=0,
        ).unsqueeze(0)
        self.decoder_pos_embed = nn.Parameter(pos_embedding, requires_grad=False)

        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_depth):
            self.decoder_layers.append(CrossAttentionBlock(encoder_dim, decoder_embed_dim, num_heads=16, mlp_ratio=4.0))

        self.decoder_norm = nn.RMSNorm(decoder_embed_dim, elementwise_affine=False)

        # Weight initialization
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)) and m.elementwise_affine is True:
                nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def mask_tokens_grid(self, mask: torch.Tensor) -> torch.Tensor:
        N = mask.size(0)
        x = self.decoder_pos_embed.masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.size(-1))
        x = x + self.mask_token

        return x

    def forward(self, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.mask_tokens_grid(mask)

        for _, layer in enumerate(self.decoder_layers):
            x = layer(x, memory)

        x = self.decoder_norm(x)

        return x


class CAPIStudent(SSLBaseNet):
    default_size = (224, 224)

    def __init__(self, input_channels: int, backbone: PreTrainEncoder, head: L2NormLinear) -> None:
        super().__init__(input_channels)
        assert isinstance(backbone, MaskedTokenOmissionMixin)
        self.backbone = backbone
        self.size = self.backbone.size

        input_size = (self.size[0] // self.backbone.max_stride, self.size[1] // self.backbone.max_stride)
        self.seq_len = input_size[0] * input_size[1]

        self.decoder = Decoder(input_size, self.backbone.embedding_size, max(8, self.backbone.num_layers // 2))
        self.head = head

    def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, x: torch.Tensor, ids_keep: torch.Tensor, ids_predict: torch.Tensor
    ) -> torch.Tensor:
        x = self.backbone.masked_encoding_omission(x, ids_keep)

        mask = masking.mask_from_indices(ids_predict, self.seq_len)
        x = self.decoder(x, mask)
        x = self.head(x.flatten(0, 1))

        return x


class CAPITeacher(SSLBaseNet):
    default_size = (224, 224)

    def __init__(self, input_channels: int, backbone: PreTrainEncoder, head: OnlineClustering) -> None:
        super().__init__(input_channels)
        assert isinstance(backbone, MaskedTokenOmissionMixin)
        self.backbone = backbone
        self.head = head

    def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, x: torch.Tensor, ids_keep: Optional[torch.Tensor], ids_predict: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)

        x = self.backbone.masked_encoding_omission(x, ids_keep)
        x = x[:, self.backbone.num_special_tokens :, :]

        (assignments, clustering_loss) = self.head(x.transpose(0, 1))

        assignments = assignments.detach().transpose(0, 1)
        row_indices = torch.arange(B).unsqueeze(1).expand_as(ids_predict)
        selected_assignments = assignments[row_indices, ids_predict]
        selected_assignments = selected_assignments.flatten(0, 1)

        return (selected_assignments, clustering_loss)
