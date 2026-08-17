"""
NaFlex preprocessing transforms, adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/pp/proj/image_text/ops_naflex.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/naflex_transforms.py

Paper "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution",
https://arxiv.org/abs/2307.06304
and
Paper "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization,
and Dense Features", https://arxiv.org/abs/2502.14786
"""

# Reference license: Apache-2.0 (both)

import math
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from birder.data.transforms.classification import AugType
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import get_training_augmentations


def get_sequence_lengths(
    image_size: tuple[int, int], patch_size: int, sizes: Optional[Sequence[int]] = None
) -> tuple[int, ...]:
    if patch_size <= 0:
        raise ValueError(f"Patch size must be positive, got {patch_size}")

    if sizes is None:
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError(f"Image size must be divisible by patch size {patch_size}, got {image_size}")

        return ((image_size[0] // patch_size) * (image_size[1] // patch_size),)

    if len(sizes) == 0:
        raise ValueError("At least one NaFlex size is required")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"NaFlex sizes must be unique, got {list(sizes)}")

    invalid_sizes = [size for size in sizes if size <= 0 or size % patch_size != 0]
    if len(invalid_sizes) > 0:
        raise ValueError(f"NaFlex sizes must be positive and divisible by patch size {patch_size}, got {invalid_sizes}")

    return tuple((size // patch_size) ** 2 for size in sizes)


def resolve_patch_grid(image_size: tuple[int, int], max_seq_len: int) -> tuple[int, int]:
    """
    Resolve a patch grid that preserves the source aspect ratio as closely as possible

    Parameters
    ----------
    image_size
        Source image size as (height, width).
    max_seq_len
        Maximum number of image patch tokens. Special tokens are not included.

    Returns
    -------
    tuple[int, int]
        Patch grid size as (height, width).
    """

    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {image_size}")
    if max_seq_len <= 0:
        raise ValueError(f"Maximum sequence length must be positive, got {max_seq_len}")
    if max_seq_len == 1:
        return (1, 1)

    # Search for the greatest uniform scale whose patch-aligned bounding grid
    # fits within the token budget. Rounding each dimension up ensures complete
    # patches at the cost of a small aspect-ratio distortion.
    lower_scale = 0.0
    upper_scale = max_seq_len / min(image_h, image_w)
    for _ in range(64):
        scale = (lower_scale + upper_scale) / 2.0
        grid_h = max(1, math.ceil(image_h * scale))
        grid_w = max(1, math.ceil(image_w * scale))
        if grid_h * grid_w <= max_seq_len:
            lower_scale = scale
        else:
            upper_scale = scale

    grid_h = max(1, math.ceil(image_h * lower_scale))
    grid_w = max(1, math.ceil(image_w * lower_scale))
    if grid_h * grid_w > max_seq_len:
        raise RuntimeError("Resolved patch grid exceeds the sequence-length limit")

    return (grid_h, grid_w)


class NativeAspectRatioResize(nn.Module):
    """
    Resize an image to a patch-aligned size under a sequence-length limit

    Parameters
    ----------
    patch_size
        Height and width of each square image patch.
    max_seq_len
        Maximum number of image patch tokens. Special tokens are not included.
    interpolation
        Interpolation method used for resizing.
    """

    def __init__(
        self, patch_size: int, max_seq_len: int, interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}")
        if max_seq_len <= 0:
            raise ValueError(f"Maximum sequence length must be positive, got {max_seq_len}")

        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.interpolation = interpolation

    def forward(self, x: Any) -> Any:
        image_size = F.get_size(x)
        grid_h, grid_w = resolve_patch_grid((image_size[0], image_size[1]), self.max_seq_len)
        output_size = (grid_h * self.patch_size, grid_w * self.patch_size)

        return F.resize(x, output_size, interpolation=self.interpolation, antialias=True)


class RandomCropWithScaleAndRelativeRatio(nn.Module):
    """
    Randomly crop an image using sampled area and relative aspect ratio

    Parameters
    ----------
    scale
        Minimum and maximum fractions of the source image area to retain.
    relative_ratio
        Minimum and maximum crop aspect ratio multipliers, relative to the source image aspect ratio.
    """

    def __init__(self, scale: tuple[float, float], relative_ratio: tuple[float, float] = (3 / 4, 4 / 3)) -> None:
        super().__init__()
        if not 0.0 < scale[0] <= scale[1] <= 1.0:
            raise ValueError(f"Scale must satisfy 0 < min <= max <= 1, got {scale}")
        if not 0.0 < relative_ratio[0] <= relative_ratio[1]:
            raise ValueError(f"Relative ratio must satisfy 0 < min <= max, got {relative_ratio}")

        self.scale = scale
        self.relative_ratio = relative_ratio

    def forward(self, x: Any) -> Any:
        image_h, image_w = F.get_size(x)
        image_ratio = image_w / image_h
        ratio = (image_ratio * self.relative_ratio[0], image_ratio * self.relative_ratio[1])
        top, left, crop_h, crop_w = v2.RandomResizedCrop.get_params(x, self.scale, ratio)

        return F.crop(x, top, left, crop_h, crop_w)


def training_preset(
    patch_size: int,
    max_seq_len: int,
    aug_type: AugType,
    level: int,
    rgv_values: RGBType,
    *,
    resize_min_scale: Optional[float] = None,
    resize_max_scale: float = 1.0,
    relative_resize_ratio: tuple[float, float] = (3 / 4, 4 / 3),
    re_prob: Optional[float] = None,
    use_grayscale: bool = False,
    ra_num_ops: int = 2,
    ra_magnitude: int = 9,
    augmix_severity: int = 3,
    clip_color_jitter_prob: float = 0.8,
    clip_gray_prob: float = 0.2,
) -> Callable[..., torch.Tensor]:
    resize = NativeAspectRatioResize(patch_size, max_seq_len)
    if aug_type == "birder" and level == 0:
        transforms: list[nn.Module] = [resize, v2.PILToTensor()]
    else:
        if resize_min_scale is None:
            if aug_type == "birder":
                resize_min_scale = 0.8 - (level * 0.05)
            elif aug_type == "clip":
                resize_min_scale = 0.8
            else:
                resize_min_scale = 0.08

        transforms = [
            v2.PILToTensor(),
            RandomCropWithScaleAndRelativeRatio(
                (resize_min_scale, resize_max_scale), relative_ratio=relative_resize_ratio
            ),
            resize,
        ]

    transforms.extend(
        get_training_augmentations(
            aug_type,
            level,
            rgv_values,
            re_prob,
            use_grayscale,
            ra_num_ops,
            ra_magnitude,
            augmix_severity,
            clip_color_jitter_prob,
            clip_gray_prob,
        )
    )

    return v2.Compose(transforms)  # type: ignore


def inference_preset(patch_size: int, max_seq_len: int, rgv_values: RGBType) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    return v2.Compose(  # type: ignore
        [
            NativeAspectRatioResize(patch_size, max_seq_len),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
