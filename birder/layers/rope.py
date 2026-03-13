"""
RoPE ViT, adapted from
https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_sincos.py
"""

# Reference license: Apache-2.0 (both)

import math
from typing import Literal
from typing import Optional

import torch
from torch import nn

RoPEStyleType = Literal["default", "axial", "centered_separate"]
RoPERotationType = Literal["standard", "interleaved"]


def _build_default_rotary_pos_embed(
    dim: int,
    temperature: float,
    grid_size: tuple[int, int],
    grid_indexing: str,
    grid_offset: int,
    pt_grid_size: Optional[tuple[int, int]],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 4 == 0
    num_bands = dim // 4
    exp = torch.arange(0, num_bands, 1, device=device) / num_bands
    bands = 1.0 / (temperature**exp)

    if pt_grid_size is None:
        pt_grid_size = grid_size

    t = [(torch.arange(s, device=device) + grid_offset) / s * p for s, p in zip(grid_size, pt_grid_size)]
    grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands
    sin_emb = pos.sin()
    cos_emb = pos.cos()

    num_spatial_dim = grid_size[0] * grid_size[1]
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)

    return (sin_emb, cos_emb)


def _build_axial_rotary_pos_embed(
    dim: int,
    temperature: float,
    grid_size: tuple[int, int],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 4 == 0
    num_bands = dim // 4
    exp = torch.arange(0, num_bands, 1, device=device) / num_bands
    bands = 1.0 / (temperature**exp)

    H, W = grid_size
    t_y = torch.arange(H, device=device, dtype=bands.dtype).view(H, 1).expand(H, W).reshape(-1)
    t_x = torch.arange(W, device=device, dtype=bands.dtype).view(1, W).expand(H, W).reshape(-1)
    angles = torch.concat((torch.outer(t_x, bands), torch.outer(t_y, bands)), dim=-1)
    sin_emb = angles.sin().repeat_interleave(2, dim=-1)
    cos_emb = angles.cos().repeat_interleave(2, dim=-1)

    return (sin_emb, cos_emb)


def _build_centered_separate_rotary_pos_embed(
    dim: int,
    temperature: float,
    grid_size: tuple[int, int],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 4 == 0
    num_bands = dim // 4
    exp = torch.arange(0, num_bands, 1, device=device) / num_bands
    bands = 1.0 / (temperature**exp)

    H, W = grid_size
    coords_h = (torch.arange(H, device=device, dtype=bands.dtype) + 0.5) / H
    coords_w = (torch.arange(W, device=device, dtype=bands.dtype) + 0.5) / W
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).reshape(-1, 2)
    coords = 2.0 * coords - 1.0

    angles = (2.0 * math.pi * coords[:, :, None]) * bands[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    sin_emb = angles.sin()
    cos_emb = angles.cos()

    return (sin_emb, cos_emb)


def validate_rope_config(
    rope_style: str,  # Uses str instead of RoPEStyleType for TorchScript compatibility
    rope_rot_type: str,  # Uses str instead of RoPERotationType for TorchScript compatibility
    grid_indexing: str,
    grid_offset: int,
    pt_grid_size: Optional[tuple[int, int]],
) -> None:
    if rope_style == "default":
        return

    if rope_style not in ("axial", "centered_separate"):
        raise ValueError(f"Unknown rope_style, got '{rope_style}'")

    if grid_indexing != "ij":
        raise ValueError(f"rope_style='{rope_style}' requires rope_grid_indexing='ij'")

    if grid_offset != 0:
        raise ValueError(f"rope_style='{rope_style}' requires rope_grid_offset=0")

    if pt_grid_size is not None:
        raise ValueError(f"rope_style='{rope_style}' does not support pt_grid_size")

    if rope_style == "axial" and rope_rot_type != "interleaved":
        raise ValueError("rope_style='axial' requires rope_rot_type='interleaved'")

    if rope_style == "centered_separate" and rope_rot_type != "standard":
        raise ValueError("rope_style='centered_separate' requires rope_rot_type='standard'")


def build_rotary_pos_embed(
    dim: int,
    temperature: float,
    grid_size: tuple[int, int],
    grid_indexing: str,
    grid_offset: int,
    pt_grid_size: Optional[tuple[int, int]],
    rope_style: str = "default",
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_style == "default":
        return _build_default_rotary_pos_embed(
            dim,
            temperature,
            grid_size=grid_size,
            grid_indexing=grid_indexing,
            grid_offset=grid_offset,
            pt_grid_size=pt_grid_size,
            device=device,
        )

    if rope_style == "axial":
        return _build_axial_rotary_pos_embed(dim, temperature, grid_size=grid_size, device=device)

    if rope_style == "centered_separate":
        return _build_centered_separate_rotary_pos_embed(dim, temperature, grid_size=grid_size, device=device)

    raise ValueError(f"Unknown rope_style, got '{rope_style}'")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.concat((-x2, x1), dim=-1)


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.size())


def apply_rotary_pos_embed(x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
    sin_emb, cos_emb = embed.tensor_split(2, dim=-1)
    if cos_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rotate_half(x) * sin_emb.unsqueeze(1).expand_as(x)

    return x * cos_emb + rotate_half(x) * sin_emb


def apply_interleaved_rotary_pos_embed(x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
    sin_emb, cos_emb = embed.tensor_split(2, dim=-1)
    if cos_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rotate_half_interleaved(x) * sin_emb.unsqueeze(1).expand_as(x)

    return x * cos_emb + rotate_half_interleaved(x) * sin_emb


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        temperature: float,
        grid_size: tuple[int, int],
        grid_indexing: Literal["ij", "xy"],
        grid_offset: int,
        pt_grid_size: Optional[tuple[int, int]] = None,
        rope_style: RoPEStyleType = "default",
        rope_rot_type: RoPERotationType = "standard",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if rope_rot_type == "standard":
            self.apply_fn = apply_rotary_pos_embed
        elif rope_rot_type == "interleaved":
            self.apply_fn = apply_interleaved_rotary_pos_embed
        else:
            raise ValueError(f"Unknown rope_rot_type, got '{rope_rot_type}'")

        validate_rope_config(rope_style, rope_rot_type, grid_indexing, grid_offset, pt_grid_size)
        sin_emb, cos_emb = build_rotary_pos_embed(
            dim,
            temperature,
            grid_size=grid_size,
            grid_indexing=grid_indexing,
            grid_offset=grid_offset,
            pt_grid_size=pt_grid_size,
            rope_style=rope_style,
            device=device,
        )
        self.pos_embed = nn.Buffer(torch.concat((sin_emb, cos_emb), dim=-1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_fn(x, self.pos_embed)
