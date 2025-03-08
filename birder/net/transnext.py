"""
TransNeXt, adapted from
https://github.com/DaiShiResearch/TransNeXt/blob/main/classification/transnext.py

Paper "TransNeXt: Robust Foveal Visual Perception for Vision Transformers", https://arxiv.org/abs/2311.17132
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torchvision.ops import StochasticDepth

from birder.kernels.load_kernel import load_swattention
from birder.model_registry import registry
from birder.net.base import DetectorBackbone

SWATTENTION = None
SWATTENTION_CUDA_NUM_THREADS = 128


@torch.no_grad()  # type: ignore[misc]
def get_relative_position_cpb(
    query_size: tuple[int, int], key_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    pretrain_size = query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)  # pylint: disable=not-callable
    axis_qw = torch.arange(query_size[1], dtype=torch.float32)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)  # pylint: disable=not-callable
    (axis_kh, axis_kw) = torch.meshgrid(axis_kh, axis_kw, indexing="ij")
    (axis_qh, axis_qw) = torch.meshgrid(axis_qh, axis_qw, indexing="ij")

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    (relative_coords_table, idx_map) = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = (
        torch.sign(relative_coords_table)
        * torch.log2(torch.abs(relative_coords_table) + 1.0)
        / torch.log2(torch.tensor(8, dtype=torch.float32))
    )

    return (idx_map, relative_coords_table)


@torch.no_grad()  # type: ignore[misc]
def get_seqlen_and_mask(input_resolution: tuple[int, int], window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    attn_map = F.unfold(
        torch.ones([1, 1, input_resolution[0], input_resolution[1]]),
        window_size,
        dilation=1,
        padding=(window_size // 2, window_size // 2),
        stride=1,
    )
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0

    return (attn_local_length, attn_mask)


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        out_features = in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=hidden_features
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (x, v) = self.fc1(x).chunk(2, dim=-1)

        (B, _, C) = x.size()
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.act(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(  # Initialize softplus(temperature) to 1/0.24
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)
        )

        # Generate sequence length scale
        self.seq_length_scale = nn.Buffer(
            torch.log(torch.tensor(input_resolution[0] * input_resolution[1])), persistent=False
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02)
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # MLP to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512)
        self.cpb_act = nn.ReLU()
        self.cpb_fc2 = nn.Linear(512, num_heads)

    def forward(
        self, x: torch.Tensor, _h: int, _w: int, relative_pos_index: torch.Tensor, relative_coords_table: torch.Tensor
    ) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        (q, k, v) = qkv.chunk(3, dim=1)

        # Use MLP to generate continuous relative positional bias
        rel_bias = (
            self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table)))
            .transpose(0, 1)[:, relative_pos_index.view(-1)]
            .view(-1, N, N)
        )

        # Calculate attention map using sequence length scaled cosine attention and query embedding
        attn = (
            (F.normalize(q, dim=-1) + self.query_embedding)
            * F.softplus(self.temperature)  # pylint: disable=not-callable
            * self.seq_length_scale
        ) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# pylint: disable=abstract-method,arguments-differ,invalid-name
class SWAttentionFunction_QK_RPB(Function):
    @staticmethod
    def forward(ctx, query, key, rpb, height, width, kernel_size):  # type: ignore
        attn_weight = SWATTENTION.qk_rpb_forward(  # type: ignore[attr-defined]
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

        (d_query, d_key, d_rpb) = SWATTENTION.qk_rpb_backward(  # type: ignore[attr-defined]
            d_attn_weight.contiguous(), query, key, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
        )

        return (d_query, d_key, d_rpb, None, None, None)


# pylint: disable=abstract-method,arguments-differ,invalid-name
class SWAttentionFunction_AV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, kernel_size):  # type: ignore
        output = SWATTENTION.av_forward(  # type: ignore[attr-defined]
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

        d_attn_weight, d_value = SWATTENTION.av_backward(  # type: ignore[attr-defined]
            d_output.contiguous(), attn_weight, value, height, width, kernel_size, SWATTENTION_CUDA_NUM_THREADS
        )

        return (d_attn_weight, d_value, None, None, None)


# pylint: disable=too-many-instance-attributes
class AggregatedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"
        assert window_size % 2 == 1, "window size must be odd"

        global SWATTENTION  # pylint: disable=global-statement
        if SWATTENTION is None and torch.jit.is_tracing() is False and torch.jit.is_scripting() is False:
            SWATTENTION = load_swattention()

        self.custom_kernel = SWATTENTION is not None

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.window_size = window_size
        self.local_len = window_size**2

        pool_h = input_resolution[0] // sr_ratio
        pool_w = input_resolution[1] // sr_ratio
        self.pool_len = pool_h * pool_w

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02)
        )
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.sr = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # MLP to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004)
        )

        # Generate padding_mask and sequence length scale
        (local_seq_length, padding_mask) = get_seqlen_and_mask(input_resolution, self.window_size)
        self.seq_length_scale = nn.Buffer(torch.log(local_seq_length + self.pool_len), persistent=False)
        self.padding_mask = nn.Buffer(padding_mask, persistent=False)

        # Dynamic local bias
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02)
        )
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(
        self, x: torch.Tensor, H: int, W: int, relative_pos_index: torch.Tensor, relative_coords_table: torch.Tensor
    ) -> torch.Tensor:
        (B, N, C) = x.size()

        # Generate queries, normalize them with L2, add query embedding,
        # and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (
            (q_norm + self.query_embedding)
            * F.softplus(self.temperature)  # pylint: disable=not-callable
            * self.seq_length_scale
        )

        # Custom kernel
        if self.custom_kernel is True and x.is_cuda is True:
            # Generate unfolded keys and values and l2-normalize them
            (k_local, v_local) = (
                self.kv(x).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2, dim=1)
            )

            # Compute local similarity
            attn_local = SWAttentionFunction_QK_RPB.apply(
                q_norm_scaled.contiguous(),
                F.normalize(k_local, dim=-1).contiguous(),
                self.relative_pos_bias_local,
                H,
                W,
                self.window_size,
            )

        # Pure PyTorch
        else:
            # Generate unfolded keys and values and l2-normalize them
            (k_local, v_local) = self.kv(x).chunk(2, dim=-1)
            k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
            kv_local = torch.concat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
            (k_local, v_local) = (
                self.unfold(kv_local)
                .reshape(B, 2 * self.num_heads, self.head_dim, self.local_len, N)
                .permute(0, 1, 4, 2, 3)
                .chunk(2, dim=1)
            )

            # Compute local similarity
            attn_local = (
                (q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) + self.relative_pos_bias_local.unsqueeze(1)
            ).masked_fill(self.padding_mask, float("-inf"))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        (k_pool, v_pool) = kv_pool.chunk(2, dim=1)

        # Use MLP to generate continuous relative positional bias for pooled features.
        pool_bias = (
            self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table)))
            .transpose(0, 1)[:, relative_pos_index.view(-1)]
            .view(-1, N, self.pool_len)
        )
        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.concat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        (attn_local, attn_pool) = torch.split(attn, [self.local_len, self.pool_len], dim=-1)

        # Custom kernel
        if self.custom_kernel is True and x.is_cuda is True:
            attn_local = (q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local
            x_local = SWAttentionFunction_AV.apply(
                attn_local.type_as(v_local), v_local.contiguous(), H, W, self.window_size
            )

        # Pure PyTorch
        else:
            x_local = (
                ((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2)
                @ v_local.transpose(-2, -1)
            ).squeeze(-2)

        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        # Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        input_resolution: tuple[int, int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: float,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if sr_ratio == 1:
            self.attn = Attention(
                dim, input_resolution, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
            )
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sr_ratio=sr_ratio,
            )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, drop=proj_drop)

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(
        self, x: torch.Tensor, H: int, W: int, relative_pos_index: torch.Tensor, relative_coords_table: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, stride: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        assert patch_size > stride, "Set larger patch_size than stride"

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(stride, stride),
            padding=(patch_size // 2, patch_size // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        (_, _, H, W) = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return (x, H, W)


class TransNeXtStage(nn.Module):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        sr_ratio: int,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
        depth: int,
    ) -> None:
        super().__init__()

        # Generate relative positional coordinate table and index for each stage
        # to compute continuous relative positional bias
        (relative_pos_index, relative_coords_table) = get_relative_position_cpb(
            query_size=input_resolution, key_size=(input_resolution[0] // sr_ratio, input_resolution[1] // sr_ratio)
        )
        self.relative_pos_index = nn.Buffer(relative_pos_index, persistent=False)
        self.relative_coords_table = nn.Buffer(relative_coords_table, persistent=False)

        self.patch_embed = OverlapPatchEmbed(
            patch_size=patch_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                AttentionBlock(
                    embed_dim,
                    num_heads=num_heads,
                    input_resolution=input_resolution,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    sr_ratio=sr_ratio,
                )
            )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        (x, H, W) = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, H, W, self.relative_pos_index, self.relative_coords_table)

        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class TransNeXt(DetectorBackbone):
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

        patch_size = 4
        sr_ratio = [8, 4, 2, 1]
        mlp_ratios = [8, 8, 4, 4]
        window_size: list[int] = self.config["window_size"]
        depth: list[int] = self.config["depth"]
        embed_dim: list[int] = self.config["embed_dim"]
        num_heads: list[int] = self.config["num_heads"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.sr_ratio = sr_ratio
        num_stages = len(depth)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = TransNeXtStage(
                input_resolution=(self.size[0] // (2 ** (i + 2)), self.size[1] // (2 ** (i + 2))),
                sr_ratio=sr_ratio[i],
                patch_size=patch_size * 2 - 1 if i == 0 else 3,
                stride=patch_size if i == 0 else 2,
                in_channels=self.input_channels if i == 0 else embed_dim[i - 1],
                embed_dim=embed_dim[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=True,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
                depth=depth[i],
            )
            return_channels.append(embed_dim[i])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dim[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        i = 0
        for m in self.body.modules():
            if isinstance(m, TransNeXtStage):
                input_resolution = (new_size[0] // (2 ** (i + 2)), new_size[1] // (2 ** (i + 2)))
                sr_ratio = self.sr_ratio[i]
                (relative_pos_index, relative_coords_table) = get_relative_position_cpb(
                    query_size=input_resolution,
                    key_size=(input_resolution[0] // sr_ratio, input_resolution[1] // sr_ratio),
                )
                m.relative_pos_index = nn.Buffer(relative_pos_index, persistent=False)
                m.relative_coords_table = nn.Buffer(relative_coords_table, persistent=False)

                for blk in m.modules():
                    if isinstance(blk, Attention):
                        blk.seq_length_scale = nn.Buffer(
                            torch.log(torch.tensor(input_resolution[0] * input_resolution[1])), persistent=False
                        )

                    elif isinstance(blk, AggregatedAttention):
                        pool_h = input_resolution[0] // sr_ratio
                        pool_w = input_resolution[1] // sr_ratio
                        blk.pool_len = pool_h * pool_w
                        blk.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))

                        (local_seq_length, padding_mask) = get_seqlen_and_mask(input_resolution, blk.window_size)
                        blk.seq_length_scale = nn.Buffer(torch.log(local_seq_length + blk.pool_len), persistent=False)
                        blk.padding_mask = nn.Buffer(padding_mask, persistent=False)

                i += 1


registry.register_alias(
    "transnext_micro",
    TransNeXt,
    config={
        "window_size": [3, 3, 3, 0],
        "depth": [2, 2, 15, 2],
        "embed_dim": [48, 96, 192, 384],
        "num_heads": [2, 4, 8, 16],
        "drop_path_rate": 0.15,
    },
)
registry.register_alias(
    "transnext_tiny",
    TransNeXt,
    config={
        "window_size": [3, 3, 3, 0],
        "depth": [2, 2, 15, 2],
        "embed_dim": [72, 144, 288, 576],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.25,
    },
)
registry.register_alias(
    "transnext_small",
    TransNeXt,
    config={
        "window_size": [3, 3, 3, 0],
        "depth": [5, 5, 22, 5],
        "embed_dim": [72, 144, 288, 576],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.45,
    },
)
registry.register_alias(
    "transnext_base",
    TransNeXt,
    config={
        "window_size": [3, 3, 3, 0],
        "depth": [5, 5, 23, 5],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [4, 8, 16, 32],
        "drop_path_rate": 0.6,
    },
)

registry.register_weights(
    "transnext_micro_il-common",
    {
        "description": "TransNeXt micro model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 48.2,
                "sha256": "2945ab2da656e944a44835c9ed04b06339cdc8e5baea4386de1f077a60ec4e0d",
            }
        },
        "net": {"network": "transnext_micro", "tag": "il-common"},
    },
)
