"""
RegionViT, adapted from
https://github.com/IBM/RegionViT/blob/master/regionvit/regionvit.py

Paper "RegionViT: Regional-to-Local Attention for Vision Transformers", https://arxiv.org/abs/2106.02689

Changes from original:
* Default size is 256 instead of 224 (window size is 8 instead of 7),
  note that the window does NOT change with resolution changes
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def convert_to_flatten_layout(
    cls_tokens: torch.Tensor, patch_tokens: torch.Tensor, ws: int
) -> tuple[torch.Tensor, Optional[torch.Tensor], int, int, int, int, int, int]:
    # Padding if added will be at the bottom right
    (B, C, H, W) = patch_tokens.size()
    (_, _, h_ks, w_ks) = cls_tokens.size()
    need_mask = False
    p_l = 0
    p_r = 0
    p_t = 0
    p_b = 0
    if H % (h_ks * ws) != 0 or W % (w_ks * ws) != 0:
        p_r = w_ks * ws - W
        p_b = h_ks * ws - H
        patch_tokens = F.pad(patch_tokens, (p_l, p_r, p_t, p_b))
        need_mask = True

    (B, C, H, W) = patch_tokens.size()
    kernel_size = (H // h_ks, W // w_ks)
    tmp = F.unfold(patch_tokens, kernel_size=kernel_size, dilation=(1, 1), padding=(0, 0), stride=kernel_size)
    patch_tokens = tmp.transpose(1, 2).reshape(-1, C, kernel_size[0] * kernel_size[1]).transpose(-2, -1)

    if need_mask is True:
        (bh_sk_s, ksks, C) = patch_tokens.size()
        h_s = H // ws
        w_s = W // ws
        mask = torch.ones(bh_sk_s // B, 1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=torch.float)
        right = torch.zeros(1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=torch.float)
        tmp = torch.zeros(ws, ws, device=patch_tokens.device, dtype=torch.float)
        tmp[0 : (ws - p_r), 0 : (ws - p_r)] = 1.0
        tmp = tmp.repeat(ws, ws)
        right[1:, 1:] = tmp
        right[0, 0] = 1
        right[0, 1:] = torch.tensor([1.0] * (ws - p_r) + [0.0] * p_r).repeat(ws).to(right.device)
        right[1:, 0] = torch.tensor([1.0] * (ws - p_r) + [0.0] * p_r).repeat(ws).to(right.device)
        bottom = torch.zeros_like(right)
        bottom[0 : ws * (ws - p_b) + 1, 0 : ws * (ws - p_b) + 1] = 1.0
        bottom_right = right.clone()
        bottom_right[0 : ws * (ws - p_b) + 1, 0 : ws * (ws - p_b) + 1] = 1.0

        mask[w_s - 1 : (h_s - 1) * w_s : w_s, ...] = right
        mask[(h_s - 1) * w_s :, ...] = bottom
        mask[-1, ...] = bottom_right
        mask = mask.repeat(B, 1, 1)
    else:
        mask = None

    cls_tokens = cls_tokens.flatten(2).transpose(-2, -1)  # (N)x(H/sxK/s)xC
    cls_tokens = cls_tokens.reshape(-1, 1, cls_tokens.size(-1))  # (NxH/sxK/s)x1xC

    out = torch.concat((cls_tokens, patch_tokens), dim=1)

    return (out, mask, p_r, p_b, B, C, H, W)


def convert_to_spatial_layout(
    out: torch.Tensor,
    B: int,
    C: int,
    H: int,
    W: int,
    kernel_size: tuple[int, int],
    mask: Optional[torch.Tensor],
    p_r: int,
    p_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cls_tokens = out[:, 0:1, ...]
    patch_tokens = out[:, 1:, ...]

    h_ks = H // kernel_size[0]
    w_ks = W // kernel_size[1]

    cls_tokens = cls_tokens.reshape(B, -1, C).transpose(-2, -1).reshape(B, C, h_ks, w_ks)
    patch_tokens = patch_tokens.transpose(1, 2).reshape((B, -1, kernel_size[0] * kernel_size[1] * C)).transpose(1, 2)
    patch_tokens = F.fold(patch_tokens, (H, W), kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))

    if mask is not None:
        if p_b > 0:
            patch_tokens = patch_tokens[:, :, :-p_b, :]
        if p_r > 0:
            patch_tokens = patch_tokens[:, :, :, :-p_r]

    return (cls_tokens, patch_tokens)


class SequentialWithTwo(nn.Sequential):
    def forward(  # pylint: disable=arguments-differ
        self, cls_tokens: torch.Tensor, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            (cls_tokens, patch_tokens) = module(cls_tokens, patch_tokens)

        return (cls_tokens, patch_tokens)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = nn.Buffer(torch.tensor(eps))
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(x.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (x - mean) / std
        out = out * self.weight + self.bias

        return out


class AttentionWithRelPos(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        attn_map_dim: tuple[int, int],
        num_cls_tokens: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_cls_tokens = num_cls_tokens
        one_dim = attn_map_dim[0]
        rel_pos_dim = 2 * one_dim - 1
        self.rel_pos = nn.Parameter(torch.zeros(num_heads, rel_pos_dim**2))
        tmp = torch.arange(rel_pos_dim**2).reshape((rel_pos_dim, rel_pos_dim))
        out = []
        offset_x = offset_y = one_dim // 2
        for y in range(one_dim):
            for x in range(one_dim):
                for dy in range(one_dim):
                    for dx in range(one_dim):
                        out.append(tmp[dy - y + offset_y, dx - x + offset_x])

        self.rel_pos_index = nn.Buffer(torch.tensor(out, dtype=torch.long))

        # Weight initialization
        nn.init.trunc_normal_(self.rel_pos, std=0.02)

    def forward(self, x: torch.Tensor, patch_attn: bool = False, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if patch_attn is True:
            rel_pos = self.rel_pos[:, self.rel_pos_index].reshape(
                self.num_heads, N - self.num_cls_tokens, N - self.num_cls_tokens
            )
            attn[:, :, self.num_cls_tokens :, self.num_cls_tokens :] = (
                attn[:, :, self.num_cls_tokens :, self.num_cls_tokens :] + rel_pos
            )

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, -65000)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_conv_type: Literal["3conv", "1conv", "linear"],
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        if patch_conv_type == "3conv":
            assert patch_size[0] == 4 and patch_size[1] == 4
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                LayerNorm2d(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                LayerNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
        elif patch_conv_type == "1conv":
            kernel_size = (2 * patch_size[0], 2 * patch_size[1])
            stride = (patch_size[0], patch_size[1])
            padding = (patch_size[0] - 1, patch_size[1] - 1)
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        elif patch_conv_type == "linear":
            kernel_size = patch_size
            stride = patch_size
            padding = (0, 0)
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError("Unknown patch_conv_type")

    def forward(self, x: torch.Tensor, extra_padding: bool = False) -> torch.Tensor:
        (_, _, H, W) = x.size()
        if extra_padding and (H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0):
            p_l = (self.patch_size[1] - W % self.patch_size[1]) // 2
            p_r = (self.patch_size[1] - W % self.patch_size[1]) - p_l
            p_t = (self.patch_size[0] - H % self.patch_size[0]) // 2
            p_b = (self.patch_size[0] - H % self.patch_size[0]) - p_t
            x = F.pad(x, (p_l, p_r, p_t, p_b))

        return self.proj(x)


class R2LAttentionFFN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        window_size: tuple[int, int],
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path: float,
        attn_drop: float,
        drop: float,
    ) -> None:
        super().__init__()

        self.norm0 = nn.LayerNorm(input_channels, eps=1e-6)
        self.norm1 = nn.LayerNorm(input_channels, eps=1e-6)
        self.attn = AttentionWithRelPos(
            input_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_map_dim=window_size,
            num_cls_tokens=1,
        )

        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(input_channels, eps=1e-6)
        self.mlp = MLP(
            input_channels, [int(input_channels * mlp_ratio), output_channels], activation_layer=nn.GELU, dropout=drop
        )

        if input_channels != output_channels:
            self.expand = nn.Sequential(
                nn.LayerNorm(input_channels, eps=1e-6),
                nn.GELU(),
                nn.Linear(input_channels, output_channels),
            )
        else:
            self.expand = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        cls_tokens = x[:, 0:1, ...]
        C = cls_tokens.size(-1)
        cls_tokens = cls_tokens.reshape(batch_size, -1, C)  # (N)x(H/sxW/s)xC
        cls_tokens = cls_tokens + self.drop_path(self.attn(self.norm0(cls_tokens)))  # (N)x(H/sxK/s)xC

        cls_tokens = cls_tokens.reshape(-1, 1, C)  # (NxH/sxK/s)x1xC

        out = torch.concat((cls_tokens, x[:, 1:, ...]), dim=1)

        out = out + self.drop_path(self.attn(self.norm1(out), patch_attn=True, mask=mask))
        identity = self.expand(out)
        out = identity + self.drop_path(self.mlp(self.norm2(out)))

        return out


class Projection(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, pool: bool) -> None:
        super().__init__()
        if pool is True:
            kernel_size = 3
            stride = 2
            padding = 1
        else:
            kernel_size = 1
            stride = 1
            padding = 0

        if input_channels == output_channels and pool is False:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                LayerNorm2d(input_channels),
                nn.GELU(),
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=(padding, padding),
                    groups=input_channels,
                ),
            )

    def forward(self, cls_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_tokens = self.proj(cls_tokens)
        patch_tokens = self.proj(patch_tokens)
        return (cls_tokens, patch_tokens)


class ConvAttStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        window_size: tuple[int, int],
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        pool: bool,
        drop_path_rate: list[float],
        attn_drop_rate: float,
        drop_rate: float,
    ):
        super().__init__()
        assert window_size[0] == window_size[1]
        self.proj = Projection(input_channels, output_channels, pool=pool)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                R2LAttentionFFN(
                    output_channels,
                    output_channels,
                    window_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_path=drop_path_rate[i],
                    attn_drop=attn_drop_rate,
                    drop=drop_rate,
                )
            )

        self.ws = window_size

    def forward(self, cls_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (cls_tokens, patch_tokens) = self.proj(cls_tokens, patch_tokens)
        (out, mask, p_r, p_b, B, C, H, W) = convert_to_flatten_layout(cls_tokens, patch_tokens, self.ws[0])
        for blk in self.blocks:
            out = blk(out, mask, B)

        (cls_tokens, patch_tokens) = convert_to_spatial_layout(out, B, C, H, W, self.ws, mask, p_r, p_b)

        return (cls_tokens, patch_tokens)


class RegionViT(DetectorBackbone):
    default_size = (256, 256)

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

        mlp_ratio = 4.0
        patch_size = (4, 4)
        patch_conv_type: Literal["3conv", "1conv", "linear"] = self.config["patch_conv_type"]
        depths: list[int] = self.config["depths"]
        embed_dims: list[int] = self.config["embed_dims"]
        num_heads: list[int] = self.config["num_heads"]
        window_size: int = self.config["window_size"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.patch_embed = PatchEmbed(patch_size, self.input_channels, embed_dims[0], patch_conv_type)
        self.cls_token = PatchEmbed(
            (patch_size[0] * window_size, patch_size[1] * window_size),
            self.input_channels,
            embed_dims[0],
            "linear",
        )

        num_stages = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = ConvAttStage(
                embed_dims[i],
                embed_dims[i + 1],
                window_size=(window_size, window_size),
                num_blocks=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                pool=i > 0,
                drop_path_rate=dpr[i],
                attn_drop_rate=0.0,
                drop_rate=0.0,
            )
            return_channels.append(embed_dims[i + 1])

        self.body = SequentialWithTwo(stages)
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True
        if unfreeze_features is True:
            for param in self.norm.parameters():
                param.requires_grad = True

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        o_x = x
        x = self.patch_embed(x)
        cls_tokens = self.cls_token(o_x, extra_padding=True)

        out = {}
        for name, module in self.body.named_children():
            (cls_tokens, x) = module(cls_tokens, x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.cls_token.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        o_x = x
        x = self.patch_embed(x)
        cls_tokens = self.cls_token(o_x, extra_padding=True)

        (cls_tokens, x) = self.body(cls_tokens, x)

        (N, C, _, _) = cls_tokens.size()
        cls_tokens = cls_tokens.reshape(N, C, -1).transpose(1, 2)
        cls_tokens = self.norm(cls_tokens)
        out = torch.mean(cls_tokens, dim=1)

        return out


registry.register_alias(
    "regionvit_t",
    RegionViT,
    config={
        "patch_conv_type": "3conv",
        "depths": [2, 2, 8, 2],
        "embed_dims": [64, 64, 128, 256, 512],
        "num_heads": [2, 4, 8, 16],
        "window_size": 8,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "regionvit_s",
    RegionViT,
    config={
        "patch_conv_type": "3conv",
        "depths": [2, 2, 8, 2],
        "embed_dims": [96, 96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 8,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "regionvit_m",
    RegionViT,
    config={
        "patch_conv_type": "1conv",
        "depths": [2, 2, 14, 2],
        "embed_dims": [96, 96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 8,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "regionvit_b",
    RegionViT,
    config={
        "patch_conv_type": "1conv",
        "depths": [2, 2, 14, 2],
        "embed_dims": [128, 128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
        "window_size": 8,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "regionvit_t_il-common",
    {
        "description": "RegionViT tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 52.3,
                "sha256": "06c9aff4086d5cb892da6dc20d7cf383e1c7e0a112948ec235615b1a4930fa5f",
            }
        },
        "net": {"network": "regionvit_t", "tag": "il-common"},
    },
)
