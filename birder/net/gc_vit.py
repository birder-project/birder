"""
GC ViT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/gcvit.py

Paper "Global Context Vision Transformers", https://arxiv.org/abs/2206.09959
"""

# Reference license: Apache-2.0

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
from birder.layers import LayerNorm2d
from birder.layers import LayerScale
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType


def window_partition(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    (B, H, W, C) = x.size()
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)

    return windows


def window_reverse(windows: torch.Tensor, window_size: tuple[int, int], H: int, W: int) -> torch.Tensor:
    C = windows.size(-1)
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)

    return x


def build_relative_position_index(window_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    coords_h = torch.arange(window_size[0], device=device)
    coords_w = torch.arange(window_size[1], device=device)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
    coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    return relative_coords.sum(-1).flatten()  # (Wh*Ww*Wh*Ww,)


def interpolate_rel_pos_bias_table(
    rel_pos_bias_table: torch.Tensor, base_window_size: tuple[int, int], new_window_size: tuple[int, int]
) -> torch.Tensor:
    if new_window_size == base_window_size:
        return rel_pos_bias_table

    (base_h, base_w) = base_window_size
    num_heads = rel_pos_bias_table.size(1)
    orig_dtype = rel_pos_bias_table.dtype
    bias_table = rel_pos_bias_table.float()
    bias_table = bias_table.view(2 * base_h - 1, 2 * base_w - 1, num_heads).permute(2, 0, 1).unsqueeze(0)
    bias_table = F.interpolate(
        bias_table,
        size=(2 * new_window_size[0] - 1, 2 * new_window_size[1] - 1),
        mode="bicubic",
        align_corners=False,
    )
    bias_table = bias_table.squeeze(0).permute(1, 2, 0).reshape(-1, num_heads)

    return bias_table.to(orig_dtype)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, squeeze_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(
            squeeze_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.scale_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.scale_act(scale)

        return x * scale


class RelPosBias(nn.Module):
    def __init__(self, window_size: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size

        bias_table = torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        self.relative_position_bias_table = nn.Parameter(bias_table)
        relative_position_index = build_relative_position_index(self.window_size, device=bias_table.device)
        self.relative_position_index = nn.Buffer(relative_position_index)

        # Weight initialization
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, window_size: tuple[int, int], dynamic_size: bool = False) -> torch.Tensor:
        if dynamic_size is False or window_size == self.window_size:
            N = self.window_size[0] * self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index].view(N, N, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

        bias_table = interpolate_rel_pos_bias_table(
            self.relative_position_bias_table,
            self.window_size,
            window_size,
        )
        relative_position_index = build_relative_position_index(window_size, device=bias_table.device)
        N = window_size[0] * window_size[1]
        relative_position_bias = bias_table[relative_position_index].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        return relative_position_bias.unsqueeze(0)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=in_channels, bias=False
        )
        self.act = nn.GELU()

        squeeze_channels = max(1, int(in_channels * 0.25))
        self.se = SqueezeExcitation(in_channels, squeeze_channels)

        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )

        if in_channels == out_channels:
            self.has_residual = True
        else:
            self.has_residual = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dw_conv(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pw_conv(x)

        if self.has_residual is True:
            x = x + residual

        return x


class FeatureBlock(nn.Module):
    def __init__(self, dim: int, levels: int) -> None:
        super().__init__()
        reductions = levels
        levels = max(1, levels)
        layers = []
        for _ in range(levels):
            layers.append(MBConvBlock(dim, dim))
            if reductions > 0:
                layers.append(nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
                reductions -= 1

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Downsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.norm1 = LayerNorm2d(in_channels)
        self.conv = MBConvBlock(in_channels, in_channels)
        self.reduction = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm2 = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.conv(x)
        x = self.reduction(x)
        x = self.norm2(x)

        return x


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.downsample = Downsample2d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.downsample(x)

        return x


class WindowAttentionGlobal(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: tuple[int, int], use_global: bool) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_global = use_global

        self.rel_pos = RelPosBias(window_size=window_size, num_heads=num_heads)
        if self.use_global is True:
            self.qkv = nn.Linear(dim, dim * 2, bias=True)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, q_global: torch.Tensor, window_size: tuple[int, int], dynamic_size: bool
    ) -> torch.Tensor:
        (B, N, C) = x.size()
        if self.use_global is True:
            kv = self.qkv(x)
            kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            (k, v) = kv.unbind(0)

            q_global = q_global.repeat(B // q_global.size(0), 1, 1, 1)
            q = q_global.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            (q, k, v) = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self.rel_pos(window_size, dynamic_size)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class GlobalContextVitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: float,
        use_global: bool,
        layer_scale: Optional[float],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionGlobal(dim, num_heads, window_size, use_global)
        if layer_scale is not None:
            self.ls1 = LayerScale(dim, layer_scale)
        else:
            self.ls1 = nn.Identity()

        self.drop_path1 = StochasticDepth(drop_path, mode="row")

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None)
        if layer_scale is not None:
            self.ls2 = LayerScale(dim, layer_scale)
        else:
            self.ls2 = nn.Identity()

        self.drop_path2 = StochasticDepth(drop_path, mode="row")
        self.dynamic_size = False

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        self.dynamic_size = dynamic_size

    def _window_attn(self, x: torch.Tensor, q_global: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
        (_, H, W, C) = x.size()

        # Pad feature maps to multiples of window size for dynamic size support
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        # Resize global query to match window size if needed
        (_, h_g, w_g, _) = q_global.size()
        if h_g != window_size[0] or w_g != window_size[1]:
            q_global = q_global.permute(0, 3, 1, 2)
            q_global = F.interpolate(q_global, size=window_size, mode="bilinear", align_corners=False)
            q_global = q_global.permute(0, 2, 3, 1)

        (_, pad_h, pad_w, _) = x.size()
        x_win = window_partition(x, window_size)
        x_win = x_win.view(-1, window_size[0] * window_size[1], C)
        attn_win = self.attn(x_win, q_global, window_size, self.dynamic_size)
        x = window_reverse(attn_win, window_size, pad_h, pad_w)

        # Unpad features
        x = x[:, :H, :W, :].contiguous()

        return x

    def forward(self, x: torch.Tensor, q_global: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self._window_attn(self.norm1(x), q_global, window_size)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class GlobalContextVitStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        feat_size: tuple[int, int],
        window_size: tuple[int, int],
        downsample: bool,
        mlp_ratio: float,
        layer_scale: Optional[float],
        stage_norm: bool,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = Downsample2d(dim, dim * 2)
            dim = dim * 2
            feat_size = (math.ceil(feat_size[0] / 2), math.ceil(feat_size[1] / 2))
        else:
            self.downsample = nn.Identity()

        self.window_size = window_size
        self.window_ratio = (max(1, feat_size[0] // window_size[0]), max(1, feat_size[1] // window_size[1]))
        self.dynamic_size = False

        feat_levels = int(math.log2(min(feat_size) / min(window_size)))
        self.global_block = FeatureBlock(dim, feat_levels)

        self.blocks = nn.ModuleList(
            [
                GlobalContextVitBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    use_global=(idx % 2 != 0),
                    layer_scale=layer_scale,
                    drop_path=drop_path[idx],
                )
                for idx in range(depth)
            ]
        )
        if stage_norm is True:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

    def _get_window_size(self, feat_size: tuple[int, int]) -> tuple[int, int]:
        if self.dynamic_size is False:
            return self.window_size

        window_h = max(1, feat_size[0] // self.window_ratio[0])
        window_w = max(1, feat_size[1] // self.window_ratio[1])
        return (window_h, window_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        window_size = self._get_window_size((x.size(2), x.size(3)))
        global_query = self.global_block(x)

        x = x.permute(0, 2, 3, 1)
        global_query = global_query.permute(0, 2, 3, 1)
        for blk in self.blocks:
            x = blk(x, global_query, window_size)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


# pylint: disable=invalid-name
class GC_ViT(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage(\d+)\.blocks\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        window_ratio: list[int] = self.config["window_ratio"]
        embed_dim: int = self.config["embed_dim"]
        mlp_ratio: float = self.config["mlp_ratio"]
        layer_scale: Optional[float] = self.config["layer_scale"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.window_ratio = window_ratio
        num_stages = len(depths)
        img_size = self.size

        # Calculate window sizes from window ratios
        window_sizes = []
        for r in window_ratio:
            window_sizes.append((max(1, img_size[0] // r), max(1, img_size[1] // r)))

        self.stem = Stem(self.input_channels, embed_dim)

        feat_size = (math.ceil(img_size[0] / 4), math.ceil(img_size[1] / 4))
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        in_dim = embed_dim
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx in range(num_stages):
            stage = GlobalContextVitStage(
                dim=in_dim,
                depth=depths[idx],
                num_heads=num_heads[idx],
                feat_size=feat_size,
                window_size=window_sizes[idx],
                downsample=idx > 0,
                mlp_ratio=mlp_ratio,
                layer_scale=layer_scale,
                stage_norm=(idx == num_stages - 1),
                drop_path=dpr[idx],
            )

            stages[f"stage{idx + 1}"] = stage
            if idx > 0:
                in_dim = in_dim * 2
                feat_size = (math.ceil(feat_size[0] / 2), math.ceil(feat_size[1] / 2))

            return_channels.append(in_dim)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = return_channels[-1]
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = embed_dim
        self.encoding_size = return_channels[-1]

        # Weight initialization
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
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        super().set_dynamic_size(dynamic_size)
        for stage in self.body.children():
            if isinstance(stage, GlobalContextVitStage):
                stage.dynamic_size = dynamic_size
                for block in stage.blocks:
                    block.set_dynamic_size(dynamic_size)

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

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        new_window_sizes = []
        for r in self.window_ratio:
            new_window_sizes.append((max(1, new_size[0] // r), max(1, new_size[1] // r)))

        feat_size = (math.ceil(new_size[0] / self.stem_stride), math.ceil(new_size[1] / self.stem_stride))
        stage_idx = 0
        for stage in self.body.children():
            if isinstance(stage, GlobalContextVitStage):
                new_window_size = new_window_sizes[stage_idx]
                if isinstance(stage.downsample, nn.Identity):
                    stage_feat_size = feat_size
                else:
                    stage_feat_size = (math.ceil(feat_size[0] / 2), math.ceil(feat_size[1] / 2))

                stage.window_size = new_window_size
                stage.window_ratio = (
                    max(1, stage_feat_size[0] // new_window_size[0]),
                    max(1, stage_feat_size[1] // new_window_size[1]),
                )
                for block in stage.blocks:
                    rel_pos = block.attn.rel_pos
                    if new_window_size == rel_pos.window_size:
                        continue

                    with torch.no_grad():
                        bias_table = interpolate_rel_pos_bias_table(
                            rel_pos.relative_position_bias_table,
                            rel_pos.window_size,
                            new_window_size,
                        )

                    rel_pos.window_size = new_window_size
                    rel_pos.relative_position_bias_table = nn.Parameter(bias_table)
                    rel_pos.relative_position_index = nn.Buffer(
                        build_relative_position_index(new_window_size, device=bias_table.device)
                    )

                feat_size = stage_feat_size
                stage_idx += 1


registry.register_model_config(
    "gc_vit_xxt",
    GC_ViT,
    config={
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 8, 16],
        "window_ratio": [32, 32, 16, 32],
        "embed_dim": 64,
        "mlp_ratio": 3.0,
        "layer_scale": None,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "gc_vit_xt",
    GC_ViT,
    config={
        "depths": [3, 4, 6, 5],
        "num_heads": [2, 4, 8, 16],
        "window_ratio": [32, 32, 16, 32],
        "embed_dim": 64,
        "mlp_ratio": 3.0,
        "layer_scale": None,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "gc_vit_t",
    GC_ViT,
    config={
        "depths": [3, 4, 19, 5],
        "num_heads": [2, 4, 8, 16],
        "window_ratio": [32, 32, 16, 32],
        "embed_dim": 64,
        "mlp_ratio": 3.0,
        "layer_scale": None,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "gc_vit_s",
    GC_ViT,
    config={
        "depths": [3, 4, 19, 5],
        "num_heads": [3, 6, 12, 24],
        "window_ratio": [32, 32, 16, 32],
        "embed_dim": 96,
        "mlp_ratio": 2.0,
        "layer_scale": 1e-5,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "gc_vit_b",
    GC_ViT,
    config={
        "depths": [3, 4, 19, 5],
        "num_heads": [4, 8, 16, 32],
        "window_ratio": [32, 32, 16, 32],
        "embed_dim": 128,
        "mlp_ratio": 2.0,
        "layer_scale": 1e-5,
        "drop_path_rate": 0.5,
    },
)
registry.register_model_config(
    "gc_vit_l",
    GC_ViT,
    config={
        "depths": [3, 4, 19, 5],
        "num_heads": [6, 12, 24, 48],
        "window_ratio": [32, 32, 16, 32],
        "embed_dim": 192,
        "mlp_ratio": 2.0,
        "layer_scale": 1e-5,
        "drop_path_rate": 0.5,
    },
)

registry.register_weights(
    "gc_vit_xxt_il-common",
    {
        "description": "GC ViT XX-Tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 47.9,
                "sha256": "5326a53903759e32178a6c2994639e6d0172faa51e1573a700f8d12b4f447c61",
            }
        },
        "net": {"network": "gc_vit_xxt", "tag": "il-common"},
    },
)
