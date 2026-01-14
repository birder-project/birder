"""
LIT v1, adapted from
https://github.com/ziplab/LIT/blob/main/classification/code_for_lit_s_m_b/models/lit.py

Paper "Less is More: Pay Less Attention in Vision Transformers", https://arxiv.org/abs/2105.14217
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def build_relative_position_index(input_resolution: tuple[int, int], device: torch.device) -> torch.Tensor:
    coords_h = torch.arange(input_resolution[0], device=device)
    coords_w = torch.arange(input_resolution[1], device=device)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += input_resolution[0] - 1
    relative_coords[:, :, 1] += input_resolution[1] - 1
    relative_coords[:, :, 0] *= 2 * input_resolution[1] - 1

    return relative_coords.sum(-1).flatten()


def interpolate_rel_pos_bias_table(
    rel_pos_bias_table: torch.Tensor, base_resolution: tuple[int, int], new_resolution: tuple[int, int]
) -> torch.Tensor:
    if new_resolution == base_resolution:
        return rel_pos_bias_table

    (base_h, base_w) = base_resolution
    num_heads = rel_pos_bias_table.size(1)
    orig_dtype = rel_pos_bias_table.dtype
    bias_table = rel_pos_bias_table.float()
    bias_table = bias_table.reshape(2 * base_h - 1, 2 * base_w - 1, num_heads).permute(2, 0, 1).unsqueeze(0)
    bias_table = F.interpolate(
        bias_table,
        size=(2 * new_resolution[0] - 1, 2 * new_resolution[1] - 1),
        mode="bicubic",
        align_corners=False,
    )
    bias_table = bias_table.squeeze(0).permute(1, 2, 0).reshape(-1, num_heads)

    return bias_table.to(orig_dtype)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor, _resolution: tuple[int, int]) -> torch.Tensor:
        return x + self.drop_path(self.mlp(self.norm(x)))


class RelPosAttention(nn.Module):
    def __init__(self, dim: int, input_resolution: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.input_resolution = input_resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.dynamic_size = False

        # Relative position bias table
        bias_table = torch.zeros((2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1), num_heads)
        self.relative_position_bias_table = nn.Parameter(bias_table)

        # Get pair-wise relative position index for each token
        relative_position_index = build_relative_position_index(input_resolution, device=bias_table.device)
        self.relative_position_index = nn.Buffer(relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Weight initialization
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _get_rel_pos_bias(self, resolution: tuple[int, int]) -> torch.Tensor:
        if self.dynamic_size is False or resolution == self.input_resolution:
            N = self.input_resolution[0] * self.input_resolution[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index].reshape(N, N, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

        bias_table = interpolate_rel_pos_bias_table(
            self.relative_position_bias_table,
            self.input_resolution,
            resolution,
        )
        relative_position_index = build_relative_position_index(resolution, device=bias_table.device)
        N = resolution[0] * resolution[1]
        relative_position_bias = bias_table[relative_position_index].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias(resolution)
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class MSABlock(nn.Module):
    def __init__(
        self, dim: int, input_resolution: tuple[int, int], num_heads: int, mlp_ratio: float, drop_path: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPosAttention(dim, input_resolution, num_heads)
        self.drop_path1 = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        self.attn.dynamic_size = dynamic_size

    def forward(self, x: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), resolution))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class DeformablePatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        kernel_size = 2

        self.offset_conv = nn.Conv2d(
            in_dim,
            2 * kernel_size * kernel_size,
            kernel_size=(kernel_size, kernel_size),
            stride=(kernel_size, kernel_size),
            padding=(0, 0),
            bias=True,
        )
        self.deform_conv = DeformConv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(kernel_size, kernel_size),
            padding=(0, 0),
            bias=True,
        )
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()

        # Initialize offsets to zero (start with regular convolution behavior)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor, resolution: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        (H, W) = resolution
        (B, _, C) = x.size()

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)

        x = self.norm(x)
        x = self.act(x)

        (B, C, H, W) = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return (x, H, W)


class IdentityDownsample(nn.Module):
    def forward(self, x: torch.Tensor, resolution: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        return (x, resolution[0], resolution[1])


class LITStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        has_msa: bool,
        downsample: bool,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = DeformablePatchMerging(in_dim, out_dim)
            resolution = (resolution[0] // 2, resolution[1] // 2)
        else:
            self.downsample = IdentityDownsample()

        blocks: list[nn.Module] = []
        for i in range(depth):
            if has_msa is True:
                blocks.append(MSABlock(out_dim, resolution, num_heads, mlp_ratio, drop_path[i]))
            else:
                blocks.append(MLPBlock(out_dim, mlp_ratio, drop_path[i]))

        self.blocks = nn.ModuleList(blocks)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        for block in self.blocks:
            if isinstance(block, MSABlock):
                block.set_dynamic_size(dynamic_size)

    def forward(self, x: torch.Tensor, input_resolution: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        (x, H, W) = self.downsample(x, input_resolution)
        for block in self.blocks:
            x = block(x, (H, W))

        return (x, H, W)


# pylint: disable=invalid-name
class LIT_v1(DetectorBackbone):
    block_group_regex = r"body\.stage(\d+)\.blocks.(\d+)"

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

        patch_size = 4
        embed_dim: int = self.config["embed_dim"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        has_msa: list[bool] = self.config["has_msa"]
        drop_path_rate: float = self.config["drop_path_rate"]

        num_stages = len(depths)

        # Patch embedding
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                padding=(0, 0),
                bias=True,
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(embed_dim),
        )

        # Stochastic depth
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        prev_dim = embed_dim
        resolution = (self.size[0] // patch_size, self.size[1] // patch_size)
        for i_stage in range(num_stages):
            in_dim = prev_dim
            out_dim = in_dim * 2 if i_stage > 0 else in_dim
            stage = LITStage(
                in_dim,
                out_dim,
                resolution,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=4.0,
                has_msa=has_msa[i_stage],
                downsample=i_stage > 0,
                drop_path=dpr[i_stage],
            )
            stages[f"stage{i_stage + 1}"] = stage

            if i_stage > 0:
                resolution = (resolution[0] // 2, resolution[1] // 2)

            prev_dim = out_dim
            return_channels.append(out_dim)

        num_features = embed_dim * (2 ** (num_stages - 1))
        self.body = nn.ModuleDict(stages)
        self.features = nn.Sequential(
            nn.LayerNorm(num_features),
            Permute([0, 2, 1]),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

        self.patch_size = patch_size

        # Weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if name.endswith("offset_conv") is True:
                    continue

                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        (B, H, W, C) = x.size()
        x = x.reshape(B, H * W, C)

        out = {}
        for name, stage in self.body.items():
            (x, H, W) = stage(x, (H, W))
            if name in self.return_stages:
                features = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                out[name] = features

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad_(False)

        for idx, stage in enumerate(self.body.values()):
            if idx >= up_to_stage:
                break

            for param in stage.parameters():
                param.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        (B, H, W, C) = x.size()
        x = x.reshape(B, H * W, C)
        for stage in self.body.values():
            (x, H, W) = stage(x, (H, W))

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        super().set_dynamic_size(dynamic_size)
        for stage in self.body.values():
            stage.set_dynamic_size(dynamic_size)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        new_patches_resolution = (new_size[0] // self.patch_size, new_size[1] // self.patch_size)

        (h, w) = new_patches_resolution
        for stage in self.body.values():
            if not isinstance(stage.downsample, IdentityDownsample):
                h = h // 2
                w = w // 2

            out_resolution = (h, w)
            for block in stage.blocks:
                if isinstance(block, MSABlock):
                    attn = block.attn
                    if out_resolution == attn.input_resolution:
                        continue

                    with torch.no_grad():
                        bias_table = interpolate_rel_pos_bias_table(
                            attn.relative_position_bias_table,
                            attn.input_resolution,
                            out_resolution,
                        )

                    attn.input_resolution = out_resolution
                    attn.relative_position_bias_table = nn.Parameter(bias_table)
                    attn.relative_position_index = nn.Buffer(
                        build_relative_position_index(out_resolution, device=bias_table.device)
                    )


registry.register_model_config(
    "lit_v1_s",
    LIT_v1,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "has_msa": [False, False, True, True],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "lit_v1_m",
    LIT_v1,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "has_msa": [False, False, True, True],
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "lit_v1_b",
    LIT_v1,
    config={
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "has_msa": [False, False, True, True],
        "drop_path_rate": 0.3,
    },
)
