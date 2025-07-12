"""
MViT v2, adapted from
https://github.com/facebookresearch/mvit
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mvitv2.py

"MViTv2: Improved Multiscale Vision Transformers for Classification and Detection",
https://arxiv.org/abs/2112.01526

Changes from original:
* Simplified MultiScaleAttention - most options removed
"""

# Reference license: Apache-2.0 (both)

import math
from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType


def pre_pool(
    x: torch.Tensor, hw_shape: tuple[int, int], has_cls_token: bool
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    (H, W) = hw_shape
    if has_cls_token is True:
        cls_tok = x[:, :, :1, :]
        x = x[:, :, 1:, :]
    else:
        cls_tok = None

    x = x.reshape(-1, H, W, x.size(-1)).permute(0, 3, 1, 2).contiguous()

    return (x, cls_tok)


def post_pool(x: torch.Tensor, num_heads: int, cls_tok: Optional[torch.Tensor]) -> tuple[torch.Tensor, tuple[int, int]]:
    hw_shape = (x.size(2), x.size(3))
    l_pooled = x.size(2) * x.size(3)
    x = x.reshape(-1, num_heads, x.shape[1], l_pooled).transpose(2, 3)
    if cls_tok is not None:
        x = torch.concat((cls_tok, x), dim=2)

    return (x, hw_shape)


def cal_rel_pos_spatial(
    attn: torch.Tensor,
    q: torch.Tensor,
    has_cls_token: bool,
    q_shape: tuple[int, int],
    k_shape: tuple[int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
) -> torch.Tensor:
    sp_idx = 1 if has_cls_token is True else 0
    (q_h, q_w) = q_shape
    (k_h, k_w) = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h, device=q.device).unsqueeze(-1) * q_h_ratio
        - torch.arange(k_h, device=q.device).unsqueeze(0) * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w, device=q.device).unsqueeze(-1) * q_w_ratio
        - torch.arange(k_w, device=q.device).unsqueeze(0) * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    rel_h = rel_pos_h[dist_h.long()]
    rel_w = rel_pos_w[dist_w.long()]

    (B, n_head, _, dim) = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, rel_h)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, rel_w)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w) + rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class SequentialWithShape(nn.Sequential):
    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor, hw_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        for module in self:
            (x, hw_shape) = module(x, hw_shape)

        return (x, hw_shape)


class PatchEmbed(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, kernel: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        (H, W) = x.shape[2:4]

        x = x.flatten(2).transpose(1, 2)

        return (x, (H, W))


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        input_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool,
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
        has_cls_token: bool,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim_out = dim_out
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_token = has_cls_token
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        dim_conv = dim_out // num_heads

        # Skip pooling with kernel and stride size of (1, 1)
        if math.prod(kernel_q) != 1 or math.prod(stride_q) != 1:
            self.pool_q = nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim_conv,
                bias=False,
            )
            self.norm_q = nn.LayerNorm(dim_conv)
        else:
            self.pool_q = None
            self.norm_q = None

        if math.prod(kernel_kv) != 1 or math.prod(stride_kv) != 1:
            self.pool_k = nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            self.norm_k = nn.LayerNorm(dim_conv)

            self.pool_v = nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            self.norm_v = nn.LayerNorm(dim_conv)
        else:
            self.pool_k = None
            self.norm_k = None
            self.pool_v = None
            self.norm_v = None

        # Relative pos embedding
        q_size_h = input_size[0] // stride_q[0]
        q_size_w = input_size[1] // stride_q[1]
        kv_size_h = input_size[0] // stride_kv[0]
        kv_size_w = input_size[1] // stride_kv[1]
        rel_sp_dim_h = 2 * max(q_size_h, kv_size_h) - 1
        rel_sp_dim_w = 2 * max(q_size_w, kv_size_w) - 1

        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim_h, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim_w, head_dim))

        # Weights initialization
        nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
        nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        (B, N, _) = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(dim=0)

        if self.pool_q is not None:
            (q, q_tok) = pre_pool(q, hw_shape, self.has_cls_token)
            q = self.pool_q(q)
            (q, q_shape) = post_pool(q, self.num_heads, q_tok)
            q = self.norm_q(q)
        else:
            q_shape = hw_shape

        if self.pool_k is not None:
            (k, k_tok) = pre_pool(k, hw_shape, self.has_cls_token)
            k = self.pool_k(k)
            (k, k_shape) = post_pool(k, self.num_heads, k_tok)
            k = self.norm_k(k)
        else:
            k_shape = hw_shape

        if self.pool_v is not None:
            (v, v_tok) = pre_pool(v, hw_shape, self.has_cls_token)
            v = self.pool_v(v)
            (v, _) = post_pool(v, self.num_heads, v_tok)
            v = self.norm_v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = cal_rel_pos_spatial(attn, q, self.has_cls_token, q_shape, k_shape, self.rel_pos_h, self.rel_pos_w)

        attn = attn.softmax(dim=-1)
        x = attn @ v
        if self.has_cls_token is True:
            x[:, :, 1:, :] = x[:, :, 1:, :] + q[:, :, 1:, :]
        else:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return (x, q_shape)


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        input_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
        has_cls_token: bool,
        dim_mul_in_att: bool,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.has_cls_token = has_cls_token
        self.dim_mul_in_att = dim_mul_in_att

        att_dim = dim_out if dim_mul_in_att is True else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            input_size=input_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            has_cls_token=has_cls_token,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(att_dim)
        self.mlp = MLP(att_dim, [int(att_dim * mlp_ratio), dim_out], activation_layer=nn.GELU, inplace=None)

        if self.dim_mul_in_att is True and self.dim != self.dim_out:
            self.proj_attn = nn.Linear(dim, dim_out)
        else:
            self.proj_attn = None

        if self.dim_mul_in_att is False and self.dim != self.dim_out:
            self.proj_mlp = nn.Linear(dim, dim_out)
        else:
            self.proj_mlp = None

        if len(stride_q) > 0 and math.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
        else:
            self.pool_skip = None

    def _shortcut_pool(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
        if self.has_cls_token is True:
            cls_tok = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            cls_tok = None

        (B, _, C) = x.size()
        (H, W) = hw_shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.pool_skip(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        if cls_tok is not None:
            x = torch.concat((cls_tok, x), dim=1)

        return x

    def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        x_norm = self.norm1(x)
        (x_block, hw_shape_new) = self.attn(x_norm, hw_shape)

        if self.proj_attn is not None:
            x = self.proj_attn(x_norm)

        if self.pool_skip is not None:
            x_res = self._shortcut_pool(x, hw_shape)
        else:
            x_res = x

        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if self.proj_mlp is not None:
            x = self.proj_mlp(x_norm)

        x = x + self.drop_path(x_mlp)

        return (x, hw_shape_new)


class MultiScaleVitStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        num_heads: int,
        input_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
        has_cls_token: bool,
        dim_mul_in_att: bool,
        drop_path: list[float],
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        if dim_mul_in_att is True:
            out_dims = (dim_out,) * depth
        else:
            out_dims = (dim,) * (depth - 1) + (dim_out,)

        for i in range(depth):
            self.blocks.append(
                MultiScaleBlock(
                    dim=dim,
                    dim_out=out_dims[i],
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    kernel_q=kernel_q,
                    kernel_kv=kernel_kv,
                    stride_q=stride_q if i == 0 else (1, 1),
                    stride_kv=stride_kv,
                    has_cls_token=has_cls_token,
                    dim_mul_in_att=dim_mul_in_att,
                    drop_path=drop_path[i],
                )
            )
            dim = out_dims[i]
            if i == 0:
                input_size = (input_size[0] // stride_q[0], input_size[1] // stride_q[1])

    def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        for blk in self.blocks:
            (x, hw_shape) = blk(x, hw_shape)

        return (x, hw_shape)


# pylint: disable=invalid-name
class MViT_v2(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.blocks\.(\d+)"

    # pylint: disable=too-many-locals
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        img_size = self.size
        stride_q = [(1, 1), (2, 2), (2, 2), (2, 2)]
        kernel_qkv = (3, 3)
        _stride_kv = (4, 4)
        depths: list[int] = self.config["depths"]
        embed_dim: int = self.config["embed_dim"]
        base_heads: int = self.config["base_heads"]
        dim_mul_in_att: bool = self.config["dim_mul_in_att"]
        use_cls_token: bool = self.config["use_cls_token"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stride_q = stride_q
        num_stages = len(depths)
        embed_dims = tuple(embed_dim * 2**i for i in range(num_stages))
        num_heads = tuple(base_heads * 2**i for i in range(num_stages))

        stride_kv = []
        for i in range(num_stages):
            if min(stride_q[i]) > 1:
                _stride_kv = (max(_stride_kv[0] // stride_q[i][0], 1), max(_stride_kv[1] // stride_q[i][1], 1))

            stride_kv.append(_stride_kv)

        self.patch_embed = PatchEmbed(
            dim_in=self.input_channels, dim_out=embed_dims[0], kernel=(7, 7), stride=(4, 4), padding=(3, 3)
        )

        if use_cls_token is True:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        input_size = (img_size[0] // 4, img_size[1] // 4)
        for i in range(num_stages):
            if dim_mul_in_att is True:
                dim_out = embed_dims[i]
            else:
                dim_out = embed_dims[min(i + 1, num_stages - 1)]

            stage = MultiScaleVitStage(
                dim=embed_dim,
                dim_out=dim_out,
                depth=depths[i],
                num_heads=num_heads[i],
                input_size=input_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                kernel_q=kernel_qkv,
                kernel_kv=kernel_qkv,
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                has_cls_token=use_cls_token,
                dim_mul_in_att=dim_mul_in_att,
                drop_path=dpr[i],
            )
            stages[f"stage{i+1}"] = stage
            return_channels.append(dim_out)
            embed_dim = dim_out
            input_size = (input_size[0] // stride_q[i][0], input_size[1] // stride_q[i][1])

        self.body = SequentialWithShape(stages)
        self.norm = nn.LayerNorm(embed_dim)
        self.return_channels = return_channels
        self.embedding_size = embed_dim
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = embed_dims[0]
        self.encoding_size = embed_dim

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (x, hw_shape) = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.concat((cls_tokens, x), dim=1)

        out = {}
        for name, module in self.body.named_children():
            (x, hw_shape) = module(x, hw_shape)
            if name in self.return_stages:
                x_inter = x
                if self.cls_token is not None:
                    x_inter = x_inter[:, 1:]

                x_inter = x_inter.reshape(x.size(0), hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2)
                out[name] = x_inter

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        B = x.size(0)

        (x, hw_shape) = self.patch_embed(x)
        x = mask_tensor(
            x.permute(0, 2, 1).reshape(B, -1, hw_shape[0], hw_shape[1]),
            mask,
            patch_factor=self.max_stride // self.stem_stride,
            mask_token=mask_token,
        )
        x = x.flatten(2).transpose(1, 2)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.concat((cls_tokens, x), dim=1)

        (x, _) = self.body(x, hw_shape)
        x = self.norm(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            if self.cls_token is not None:
                features = x[:, 1:]
            else:
                features = x

            features = features.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2)
            result["features"] = features

        if return_keys in ("all", "embedding"):
            if self.cls_token is not None:
                result["embedding"] = x[:, 0]
            else:
                result["embedding"] = x.mean(1)

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        (x, hw_shape) = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.concat((cls_tokens, x), dim=1)

        (x, _) = self.body(x, hw_shape)
        x = self.norm(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            x = x.mean(1)

        return x

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        input_size = (new_size[0] // 4, new_size[1] // 4)
        for i, module in enumerate(self.body.children()):
            if isinstance(module, MultiScaleVitStage):
                idx = 0
                for m in module.modules():
                    if isinstance(m, MultiScaleBlock):
                        q_size_h = input_size[0] // m.attn.stride_q[0]
                        q_size_w = input_size[1] // m.attn.stride_q[1]
                        kv_size_h = input_size[0] // m.attn.stride_kv[0]
                        kv_size_w = input_size[1] // m.attn.stride_kv[1]
                        rel_sp_dim_h = 2 * max(q_size_h, kv_size_h) - 1
                        rel_sp_dim_w = 2 * max(q_size_w, kv_size_w) - 1

                        rel_pos_h = m.attn.rel_pos_h
                        rel_pos_h_resized = F.interpolate(
                            rel_pos_h.reshape(1, rel_pos_h.shape[0], -1).permute(0, 2, 1),
                            size=rel_sp_dim_h,
                            mode="linear",
                        )
                        rel_pos_w = m.attn.rel_pos_w
                        rel_pos_w_resized = F.interpolate(
                            rel_pos_w.reshape(1, rel_pos_w.shape[0], -1).permute(0, 2, 1),
                            size=rel_sp_dim_w,
                            mode="linear",
                        )

                        m.attn.rel_pos_h = nn.Parameter(rel_pos_h_resized.reshape(-1, rel_sp_dim_h).permute(1, 0))
                        m.attn.rel_pos_w = nn.Parameter(rel_pos_w_resized.reshape(-1, rel_sp_dim_w).permute(1, 0))

                        if idx == 0:
                            input_size = (input_size[0] // self.stride_q[i][0], input_size[1] // self.stride_q[i][1])

                        idx += 1


registry.register_model_config(
    "mvit_v2_t",
    MViT_v2,
    config={
        "depths": [1, 2, 5, 2],
        "embed_dim": 96,
        "base_heads": 1,
        "dim_mul_in_att": True,
        "use_cls_token": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "mvit_v2_s",
    MViT_v2,
    config={
        "depths": [1, 2, 11, 2],
        "embed_dim": 96,
        "base_heads": 1,
        "dim_mul_in_att": True,
        "use_cls_token": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "mvit_v2_b",
    MViT_v2,
    config={
        "depths": [2, 3, 16, 3],
        "embed_dim": 96,
        "base_heads": 1,
        "dim_mul_in_att": True,
        "use_cls_token": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "mvit_v2_l",
    MViT_v2,
    config={
        "depths": [2, 6, 36, 4],
        "embed_dim": 144,
        "base_heads": 2,
        "dim_mul_in_att": False,
        "use_cls_token": False,
        "drop_path_rate": 0.5,
    },
)
registry.register_model_config(
    "mvit_v2_t_cls",
    MViT_v2,
    config={
        "depths": [1, 2, 5, 2],
        "embed_dim": 96,
        "base_heads": 1,
        "dim_mul_in_att": True,
        "use_cls_token": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "mvit_v2_s_cls",
    MViT_v2,
    config={
        "depths": [1, 2, 11, 2],
        "embed_dim": 96,
        "base_heads": 1,
        "dim_mul_in_att": True,
        "use_cls_token": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "mvit_v2_b_cls",
    MViT_v2,
    config={
        "depths": [2, 3, 16, 3],
        "embed_dim": 96,
        "base_heads": 1,
        "dim_mul_in_att": True,
        "use_cls_token": True,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "mvit_v2_l_cls",
    MViT_v2,
    config={
        "depths": [2, 6, 36, 4],
        "embed_dim": 144,
        "base_heads": 2,
        "dim_mul_in_att": True,
        "use_cls_token": True,
        "drop_path_rate": 0.5,
    },
)
registry.register_model_config(
    "mvit_v2_h_cls",
    MViT_v2,
    config={
        "depths": [4, 8, 60, 8],
        "embed_dim": 192,
        "base_heads": 3,
        "dim_mul_in_att": True,
        "use_cls_token": True,
        "drop_path_rate": 0.8,
    },
)

registry.register_weights(
    "mvit_v2_t_il-all",
    {
        "url": "https://huggingface.co/birder-project/mvit_v2_t_il-all/resolve/main",
        "description": "MViT v2 tiny model trained on the il-all dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 91.2,
                "sha256": "2f15635de9a9020b8e7123d3a342c072212fa720e65a5cae910efd202d427298",
            }
        },
        "net": {"network": "mvit_v2_t", "tag": "il-all"},
    },
)
