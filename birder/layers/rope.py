"""
RoPE ViT, adapted from
https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_sincos.py
"""

# Reference license: Apache-2.0 (both)

import math
from collections.abc import Callable
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

    if grid_indexing == "xy":
        grid_size = (grid_size[1], grid_size[0])
        if pt_grid_size is not None:
            pt_grid_size = (pt_grid_size[1], pt_grid_size[0])

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
    dim: int, temperature: float, grid_size: tuple[int, int], device: Optional[torch.device] = None
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
    shift_coords: Optional[float] = None,
    jitter_coords: Optional[float] = None,
    rescale_coords: Optional[float] = None,
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

    # Independently shift both axes by a uniform value in [-shift, shift]
    if shift_coords is not None:
        shift_hw = torch.empty(2, device=device, dtype=bands.dtype).uniform_(-shift_coords, shift_coords)
        coords += shift_hw[None, :]

    # Independently scale both axes by a log-uniform value in [1 / jitter, jitter]
    if jitter_coords is not None:
        jitter_max = math.log(jitter_coords)
        jitter_hw = torch.empty(2, device=device, dtype=bands.dtype).uniform_(-jitter_max, jitter_max).exp()
        coords *= jitter_hw[None, :]

    # Scale both axes by the same log-uniform value in [1 / rescale, rescale]
    if rescale_coords is not None:
        rescale_max = math.log(rescale_coords)
        rescale = torch.empty(1, device=device, dtype=bands.dtype).uniform_(-rescale_max, rescale_max).exp()
        coords *= rescale

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
    shift_coords: Optional[float] = None,
    jitter_coords: Optional[float] = None,
    rescale_coords: Optional[float] = None,
) -> None:
    if shift_coords is not None:
        if rope_style != "centered_separate":
            raise ValueError("shift_coords is only supported for rope_style='centered_separate'")
        if shift_coords < 0.0:
            raise ValueError(f"shift_coords must be greater than or equal to 0, got {shift_coords}")

    if jitter_coords is not None:
        if rope_style != "centered_separate":
            raise ValueError("jitter_coords is only supported for rope_style='centered_separate'")
        if jitter_coords < 1.0:
            raise ValueError(f"jitter_coords must be greater than or equal to 1, got {jitter_coords}")

    if rescale_coords is not None:
        if rope_style != "centered_separate":
            raise ValueError("rescale_coords is only supported for rope_style='centered_separate'")
        if rescale_coords < 1.0:
            raise ValueError(f"rescale_coords must be greater than or equal to 1, got {rescale_coords}")

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
    shift_coords: Optional[float] = None,
    jitter_coords: Optional[float] = None,
    rescale_coords: Optional[float] = None,
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
        return _build_centered_separate_rotary_pos_embed(
            dim,
            temperature,
            grid_size=grid_size,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
            device=device,
        )

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


def get_rope_apply_fn(
    rope_rot_type: RoPERotationType,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if rope_rot_type == "standard":
        return apply_rotary_pos_embed

    if rope_rot_type == "interleaved":
        return apply_interleaved_rotary_pos_embed

    raise ValueError(f"Unknown rope_rot_type, got '{rope_rot_type}'")


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
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        validate_rope_config(
            rope_style,
            rope_rot_type,
            grid_indexing,
            grid_offset,
            pt_grid_size,
            shift_coords,
            jitter_coords,
            rescale_coords,
        )

        self.apply_fn = get_rope_apply_fn(rope_rot_type)
        self.dim = dim
        self.temperature = temperature
        self.grid_size = grid_size
        self.grid_indexing = grid_indexing
        self.grid_offset = grid_offset
        self.pt_grid_size = pt_grid_size
        self.rope_style = rope_style
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.pos_embed = nn.Buffer(self._build_pos_embed(grid_size, device=device), persistent=False)

    def _build_pos_embed(
        self,
        grid_size: tuple[int, int],
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        sin_emb, cos_emb = build_rotary_pos_embed(
            self.dim,
            self.temperature,
            grid_size=grid_size,
            grid_indexing=self.grid_indexing,
            grid_offset=self.grid_offset,
            pt_grid_size=self.pt_grid_size,
            rope_style=self.rope_style,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
            device=device,
        )

        return torch.concat((sin_emb, cos_emb), dim=-1)

    def get_pos_embed(self, grid_size: tuple[int, int]) -> torch.Tensor:
        if self.training is False or (
            self.shift_coords is None and self.jitter_coords is None and self.rescale_coords is None
        ):
            if grid_size == self.grid_size:
                return self.pos_embed

            pos_embed = self._build_pos_embed(grid_size, device=self.pos_embed.device)
        else:
            pos_embed = self._build_pos_embed(
                grid_size,
                shift_coords=self.shift_coords,
                jitter_coords=self.jitter_coords,
                rescale_coords=self.rescale_coords,
                device=self.pos_embed.device,
            )

        return pos_embed.to(dtype=self.pos_embed.dtype)

    def set_grid_size(self, grid_size: tuple[int, int]) -> None:
        if grid_size == self.grid_size:
            return

        pos_embed = self._build_pos_embed(grid_size, device=self.pos_embed.device)
        self.pos_embed = pos_embed.to(dtype=self.pos_embed.dtype)
        self.grid_size = grid_size

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_embed = self.get_pos_embed(self.grid_size)
        return (self.apply_fn(q, pos_embed), self.apply_fn(k, pos_embed))
