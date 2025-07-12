"""
BiFormer, adapted from
https://github.com/rayleizhu/BiFormer/blob/public_release/models/biformer.py

"BiFormer: Vision Transformer with Bi-Level Routing Attention", https://arxiv.org/abs/2303.08810

Changes from original:
* All attention types are in (B, C, H, W)
* Using the newer Bi-Level Routing Attention implementation
* Dynamic n_win size (image size // 32)
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.layers import LayerNorm2d
from birder.layers import LayerScale2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def _grid2seq(x: torch.Tensor, region_size: tuple[int, int], num_heads: int) -> tuple[torch.Tensor, int, int]:
    (B, C, H, W) = x.size()
    region_h = H // region_size[0]
    region_w = W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum("bmdhpwq->bmhwpqd", x).flatten(2, 3).flatten(-3, -2)  # (bs, n_head, n_region, reg_size, head_dim)

    return (x, region_h, region_w)


def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: tuple[int, int]) -> torch.Tensor:
    (bs, n_head, _, _, head_dim) = x.size()
    x = x.view(bs, n_head, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum("bmhwpqd->bmdhpwq", x).reshape(
        bs, n_head * head_dim, region_h * region_size[0], region_w * region_size[1]
    )

    return x


# pylint: disable=too-many-locals
def regional_routing_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    region_graph: torch.Tensor,  # LongTensor
    region_size: tuple[int, int],
    auto_pad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_region_size = region_size
    (bs, n_head, q_nregion, topk) = region_graph.size()

    # Pad to deal with any input size
    q_pad_b = 0
    q_pad_r = 0
    kv_pad_b = 0
    kv_pad_r = 0
    if auto_pad is True:
        (_, _, h_q, w_q) = query.size()
        q_pad_b = (region_size[0] - h_q % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - w_q % region_size[1]) % region_size[1]
        if q_pad_b > 0 or q_pad_r > 0:
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))

        (_, _, h_k, w_k) = key.size()
        kv_pad_b = (kv_region_size[0] - h_k % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - w_k % kv_region_size[1]) % kv_region_size[1]
        if kv_pad_r > 0 or kv_pad_b > 0:
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))
    else:
        h_q = None
        w_q = None
        h_k = None
        w_k = None

    # To sequence format
    (query, q_region_h, q_region_w) = _grid2seq(query, region_size=region_size, num_heads=n_head)
    (key, _, _) = _grid2seq(key, region_size=kv_region_size, num_heads=n_head)
    (value, _, _) = _grid2seq(value, region_size=kv_region_size, num_heads=n_head)

    # Gather key and values
    (bs, n_head, kv_nregion, kv_region_size, head_dim) = key.size()
    broadcasted_region_graph = region_graph.view(bs, n_head, q_nregion, topk, 1, 1).expand(
        -1, -1, -1, -1, kv_region_size, head_dim
    )
    key_g = torch.gather(
        key.view(bs, n_head, 1, kv_nregion, kv_region_size, head_dim).expand(-1, -1, query.size(2), -1, -1, -1),
        dim=3,
        index=broadcasted_region_graph,
    )  # (bs, n_head, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(
        value.view(bs, n_head, 1, kv_nregion, kv_region_size, head_dim).expand(-1, -1, query.size(2), -1, -1, -1),
        dim=3,
        index=broadcasted_region_graph,
    )  # (bs, n_head, q_nregion, topk, kv_region_size, head_dim)

    # Token-to-token attention
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ value_g.flatten(-3, -2)

    # To grid format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # Remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :h_q, :w_q]

    return (output, attn)


class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, n_win_h: int, n_win_w: int, topk: int, side_dwconv: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = dim**-0.5
        assert dim % num_heads == 0

        self.lepe = nn.Conv2d(
            dim,
            dim,
            kernel_size=(side_dwconv, side_dwconv),
            stride=(1, 1),
            padding=(side_dwconv // 2, side_dwconv // 2),
            groups=dim,
        )
        self.topk = topk
        self.n_win_h = n_win_h
        self.n_win_w = n_win_w

        self.qkv_linear = nn.Conv2d(dim, 3 * dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_linear = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (_, _, H, W) = x.size()
        region_size = (H // self.n_win_h, W // self.n_win_w)

        # Linear projection
        qkv = self.qkv_linear(x)
        (q, k, v) = qkv.chunk(3, dim=1)

        # Region-to-region routing
        q_r = F.avg_pool2d(  # pylint: disable=not-callable
            q.detach(), kernel_size=region_size, stride=region_size, ceil_mode=True, count_include_pad=False
        )
        k_r = F.avg_pool2d(  # pylint: disable=not-callable
            k.detach(), kernel_size=region_size, stride=region_size, ceil_mode=True, count_include_pad=False
        )
        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # (n, (hw), c)
        k_r = k_r.flatten(2, 3)  # (n, c, (hw))
        a_r = q_r @ k_r
        (_, idx_r) = torch.topk(a_r, k=self.topk, dim=-1)
        idx_r = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # Token to token attention
        (output, _) = regional_routing_attention_torch(
            q, k, v, scale=self.scale, region_graph=idx_r, region_size=region_size, auto_pad=True
        )

        output = output + self.lepe(v)
        output = self.output_linear(output)

        return output


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(B, C, H, W).permute(0, 3, 1, 2)

        return x


class AttentionLePE(nn.Module):
    """
    Attention with Locally-enhanced Positional Encoding
    """

    def __init__(
        self, dim: int, num_heads: int, qkv_bias: bool, side_dwconv: int, attn_drop: float, proj_drop: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(
            dim,
            dim,
            kernel_size=(side_dwconv, side_dwconv),
            stride=(1, 1),
            padding=(side_dwconv // 2, side_dwconv // 2),
            groups=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.size()
        (q, k, v) = self.qkv(x).chunk(3, dim=1)

        attn = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2) @ k.view(
            B, self.num_heads, self.head_dim, H * W
        )
        attn = torch.softmax(attn * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        output = attn @ v.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = output + self.lepe(v)

        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class BiFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_win_h: int,
        n_win_w: int,
        topk: int,
        mlp_ratio: float,
        side_dwconv: int,
        layer_scale_init_value: Optional[float],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)
        self.norm1 = LayerNorm2d(dim, eps=1e-6)
        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win_h=n_win_h, n_win_w=n_win_w, topk=topk, side_dwconv=side_dwconv
            )
        elif topk == -1:
            self.attn = Attention(dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0)
        elif topk == -2:
            self.attn = AttentionLePE(
                dim, num_heads=8, qkv_bias=False, side_dwconv=side_dwconv, attn_drop=0.0, proj_drop=0.0
            )
        elif topk == 0:
            self.attn = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                nn.Conv2d(dim, dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=dim),
                nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            )
        else:
            raise ValueError(f"topk={topk} not supported")

        self.norm2 = LayerNorm2d(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)

        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x


class BiFormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        n_win_h: int,
        n_win_w: int,
        topk: int,
        mlp_ratio: float,
        side_dwconv: int,
        layer_scale_init_value: Optional[float],
        drop_path: list[float],
        depth: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(out_dim),
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        layers = []
        for i in range(depth):
            layers.append(
                BiFormerBlock(
                    out_dim,
                    num_heads=num_heads,
                    n_win_h=n_win_h,
                    n_win_w=n_win_w,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_path=drop_path[i],
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class BiFormer(DetectorBackbone):
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

        head_dim = 32
        topks = [1, 4, 16, -2]
        mlp_ratios = [3, 3, 3, 3]
        side_dwconv = 5
        depths: list[int] = self.config["depths"]
        embed_dims: list[int] = self.config["embed_dims"]
        qk_dims: list[int] = self.config["qk_dims"]
        layer_scale_init_value: Optional[float] = self.config["layer_scale_init_value"]
        drop_path_rate: float = self.config["drop_path_rate"]
        n_win_h = self.size[0] // 32
        n_win_w = self.size[1] // 32

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                embed_dims[0] // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.GELU,
                inplace=None,
            ),
            Conv2dNormActivation(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=None,
            ),
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)
        n_heads = [dim // head_dim for dim in qk_dims]

        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = BiFormerStage(
                prev_dim,
                embed_dims[i],
                num_heads=n_heads[i],
                n_win_h=n_win_h,
                n_win_w=n_win_w,
                topk=topks[i],
                mlp_ratio=mlp_ratios[i],
                side_dwconv=side_dwconv,
                layer_scale_init_value=layer_scale_init_value,
                drop_path=dpr[i],
                depth=depths[i],
                downsample=i > 0,
            )
            return_channels.append(embed_dims[i])
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(embed_dims[-1]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        n_win_h = new_size[0] // 32
        n_win_w = new_size[1] // 32
        for m in self.modules():
            if isinstance(m, BiLevelRoutingAttention):
                m.n_win_h = n_win_h
                m.n_win_w = n_win_w


registry.register_model_config(
    "biformer_t",
    BiFormer,
    config={
        "depths": [2, 2, 8, 2],
        "embed_dims": [64, 128, 256, 512],
        "qk_dims": [64, 128, 256, 512],
        "layer_scale_init_value": None,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "biformer_s",
    BiFormer,
    config={
        "depths": [4, 4, 18, 4],
        "embed_dims": [64, 128, 256, 512],
        "qk_dims": [64, 128, 256, 512],
        "layer_scale_init_value": None,
        "drop_path_rate": 0.15,
    },
)
registry.register_model_config(
    "biformer_b",
    BiFormer,
    config={
        "depths": [4, 4, 18, 4],
        "embed_dims": [96, 192, 384, 768],
        "qk_dims": [96, 192, 384, 768],
        "layer_scale_init_value": None,
        "drop_path_rate": 0.4,
    },
)

registry.register_weights(
    "biformer_s_il-all",
    {
        "url": "https://huggingface.co/birder-project/biformer_s_il-all/resolve/main",
        "description": "BiFormer small model trained on the il-all dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 96.8,
                "sha256": "ecce546d5adde5cc335a98245b9cc7fddb0ba43628b0d105d974ae6537cbacf7",
            }
        },
        "net": {"network": "biformer_s", "tag": "il-all"},
    },
)
