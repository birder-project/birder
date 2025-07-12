"""
DaViT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/davit.py

Paper "DaViT: Dual Attention Vision Transformers", https://arxiv.org/abs/2204.03645

Changes from original:
* Window size based on image size (image size // 32)
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

from birder.common.masking import mask_tensor
from birder.layers import LayerNorm2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType


def window_partition(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    (B, H, W, C) = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)

    return windows


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int], act: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=dim,
        )
        if act is True:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.act(self.proj(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.even_k = kernel_size % 2 == 0
        if self.even_k is True:
            padding = (0, 0)
        else:
            padding = (kernel_size // 2, kernel_size // 2)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(2, 2), padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (_, _, H, W) = x.shape
        x = self.norm(x)
        if self.even_k is True:
            (k_h, k_w) = self.conv.kernel_size
            pad_r = (k_w - W % k_w) % k_w
            pad_b = (k_h - H % k_h) % k_h
            x = F.pad(x, (0, pad_r, 0, pad_b))

        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        k = k * self.scale
        attn = k.transpose(-1, -2) @ v
        attn = attn.softmax(dim=-1)
        x = (attn @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class ChannelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path: float,
        cpe_act: bool = False,
    ) -> None:
        super().__init__()
        self.cpe1 = ConvPosEnc(dim=dim, kernel_size=(3, 3), act=cpe_act)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.cpe2 = ConvPosEnc(dim=dim, kernel_size=(3, 3), act=cpe_act)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.shape
        x = self.cpe1(x).flatten(2).transpose(1, 2)

        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # pylint: disable=not-callable
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path: float,
        cpe_act: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.cpe1 = ConvPosEnc(dim=dim, kernel_size=(3, 3), act=cpe_act)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.cpe2 = ConvPosEnc(dim=dim, kernel_size=(3, 3), act=cpe_act)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.shape

        shortcut = self.cpe1(x).flatten(2).transpose(1, 2)

        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        (_, Hp, Wp, _) = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        Ca = attn_windows.shape[-1]
        x = attn_windows.view(
            -1, Hp // self.window_size[0], Wp // self.window_size[1], self.window_size[0], self.window_size[1], Ca
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Hp, Wp, Ca)

        # In case pad_r > 0 or pad_b > 0
        x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class DaViTStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample: bool,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path_rates: list[float],
        cpe_act: bool,
        down_kernel_size: int,
    ) -> None:
        super().__init__()
        if downsample:
            self.downsample = Downsample(in_channels, out_channels, kernel_size=down_kernel_size)
        else:
            self.downsample = nn.Identity()

        stage_blocks = []
        for block_idx in range(depth):
            stage_blocks.append(
                SpatialBlock(
                    out_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path_rates[block_idx],
                    cpe_act=cpe_act,
                )
            )
            stage_blocks.append(
                ChannelBlock(
                    out_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path_rates[block_idx],
                    cpe_act=cpe_act,
                )
            )

        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class DaViT(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.blocks.(\d+)"

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
        qkv_bias = True
        down_kernel_size = 2
        window_size = (int(self.size[0] / (2**5)), int(self.size[1] / (2**5)))
        depths: list[int] = self.config["depths"]
        dims: list[int] = self.config["dims"]
        heads: list[int] = self.config["heads"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = Stem(self.input_channels, dims[0])

        num_stages = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        in_channels = dims[0]
        for stage_idx in range(num_stages):
            out_channels = dims[stage_idx]
            stages[f"stage{stage_idx+1}"] = DaViTStage(
                in_channels,
                out_channels,
                depth=depths[stage_idx],
                downsample=stage_idx > 0,
                num_heads=heads[stage_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rates=dpr[stage_idx],
                cpe_act=False,
                down_kernel_size=down_kernel_size,
            )
            return_channels.append(out_channels)
            in_channels = out_channels

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(dims[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = dims[-1]
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = dims[0]
        self.encoding_size = dims[-1]

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
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

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        x = self.stem(x)
        x = mask_tensor(x, mask, patch_factor=self.max_stride // self.stem_stride, mask_token=mask_token)
        x = self.body(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            result["features"] = x
        if return_keys in ("all", "embedding"):
            result["embedding"] = self.features(x)

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:  # pylint:disable=useless-parent-delegation
        super().set_dynamic_size(dynamic_size)
        # No action taken, just run with constant window form initialization

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        new_window_size = (int(new_size[0] / (2**5)), int(new_size[1] / (2**5)))
        for m in self.body.modules():
            if isinstance(m, SpatialBlock):
                m.window_size = new_window_size


registry.register_model_config(
    "davit_tiny",
    DaViT,
    config={"depths": [1, 1, 3, 1], "dims": [96, 192, 384, 768], "heads": [3, 6, 12, 24], "drop_path_rate": 0.1},
)
registry.register_model_config(
    "davit_small",
    DaViT,
    config={"depths": [1, 1, 9, 1], "dims": [96, 192, 384, 768], "heads": [3, 6, 12, 24], "drop_path_rate": 0.2},
)
registry.register_model_config(
    "davit_base",
    DaViT,
    config={"depths": [1, 1, 9, 1], "dims": [128, 256, 512, 1024], "heads": [4, 8, 16, 32], "drop_path_rate": 0.4},
)
registry.register_model_config(
    "davit_large",
    DaViT,
    config={"depths": [1, 1, 9, 1], "dims": [192, 384, 768, 1536], "heads": [6, 12, 24, 48], "drop_path_rate": 0.4},
)
registry.register_model_config(
    "davit_huge",
    DaViT,
    config={"depths": [1, 1, 9, 1], "dims": [256, 512, 1024, 2048], "heads": [8, 16, 32, 64], "drop_path_rate": 0.5},
)
registry.register_model_config(
    "davit_giant",
    DaViT,
    config={"depths": [1, 1, 12, 3], "dims": [384, 768, 1536, 3072], "heads": [12, 24, 48, 96], "drop_path_rate": 0.5},
)

registry.register_weights(
    "davit_tiny_il-all",
    {
        "url": "https://huggingface.co/birder-project/davit_tiny_il-all/resolve/main",
        "description": "DaViT tiny model trained on the il-all dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 107.0,
                "sha256": "ba42050975d56527b61eac6faf0de77bb6aa0467fd676ef3556654cd967146a4",
            }
        },
        "net": {"network": "davit_tiny", "tag": "il-all"},
    },
)
