import random
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import v2

from birder.data.transforms.classification import RGBType


class ResizeWithRandomInterpolation(nn.Module):
    def __init__(
        self, size: Optional[int] | tuple[int, int], max_size: Optional[int], interpolation: list[v2.InterpolationMode]
    ) -> None:
        super().__init__()
        self.transform = []
        for interp in interpolation:
            self.transform.append(
                v2.Resize(
                    size,
                    interpolation=interp,
                    max_size=max_size,
                    antialias=True,
                )
            )

    def forward(self, *x: Any) -> torch.Tensor:
        t = random.choice(self.transform)
        return t(x)


def get_birder_augment(
    size: tuple[int, int],
    level: int,
    fill_value: list[float],
    dynamic_size: bool,
    multiscale: bool,
    max_size: Optional[int],
) -> Callable[..., torch.Tensor]:
    if dynamic_size is True:
        target_size: Optional[int] | tuple[int, int] = min(size)
    elif max_size is not None:
        target_size = None
    else:
        target_size = size

    transformations = []

    # Base augmentations
    if level >= 1:
        if dynamic_size is False and multiscale is False:
            transformations.extend(
                [
                    v2.RandomChoice(
                        [
                            v2.ScaleJitter(
                                target_size=size, scale_range=(max(0.1, 0.5 - (0.08 * level)), 2), antialias=True
                            ),
                            v2.RandomZoomOut(fill_value, side_range=(1, 3 + level * 0.1), p=0.5),
                        ]
                    ),
                ]
            )

    if level >= 3:
        transformations.extend(
            [
                v2.RandomIoUCrop(),
                v2.ClampBoundingBoxes(),
            ]
        )

    # Resize
    if multiscale is True:
        transformations.append(
            v2.RandomShortestSize(
                min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=max_size or 1333
            ),
        )
    else:
        transformations.append(
            ResizeWithRandomInterpolation(
                target_size, max_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
            ),
        )

    # Classification style augmentations
    if level >= 4:
        transformations.extend(
            [
                v2.RandomChoice(
                    [
                        v2.ColorJitter(
                            brightness=0.1 + (0.0125 * level),
                            contrast=0.0 + (0.015 * level),
                            hue=max(0, -0.025 + (level * 0.01)),
                        ),
                        v2.RandomPhotometricDistort(p=1.0),
                        v2.Identity(),
                    ]
                ),
            ]
        )

    if level >= 6:
        transformations.extend(
            [
                v2.RandomChoice(
                    [
                        v2.RandomGrayscale(p=0.5),
                        v2.RandomSolarize(255 - (10 * level), p=0.5),
                    ]
                ),
            ]
        )

    transformations.extend(
        [
            v2.RandomHorizontalFlip(0.5),
            v2.SanitizeBoundingBoxes(),
        ]
    )

    return v2.Compose(transformations)  # type: ignore


AugType = Literal["birder", "lsj", "multiscale", "ssd", "ssdlite", "detr"]


def training_preset(
    size: tuple[int, int],
    aug_type: AugType,
    level: int,
    rgv_values: RGBType,
    dynamic_size: bool = False,
    multiscale: bool = False,
    max_size: Optional[int] = None,
) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]
    fill_value = [255 * v for v in mean]
    if dynamic_size is True:
        target_size: Optional[int] | tuple[int, int] = min(size)
    elif max_size is not None:
        target_size = None
    else:
        target_size = size

    if aug_type == "birder":
        if 0 > level or level > 10:
            raise ValueError("Unsupported aug level")

        return v2.Compose(  # type:ignore
            [
                v2.ToImage(),
                get_birder_augment(size, level, fill_value, dynamic_size, multiscale, max_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    if aug_type == "lsj":
        return v2.Compose(  # type: ignore
            [
                v2.ToImage(),
                v2.ScaleJitter(target_size=size, scale_range=(0.1, 2), antialias=True),
                ResizeWithRandomInterpolation(  # Supposed to be FixedSizeCrop
                    target_size, max_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    if aug_type == "multiscale":
        return v2.Compose(  # type: ignore
            [
                v2.ToImage(),
                v2.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=max_size or 1333
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    if aug_type == "ssd":
        return v2.Compose(  # type: ignore
            [
                v2.ToImage(),
                v2.RandomPhotometricDistort(),
                v2.RandomZoomOut(fill_value),
                v2.RandomIoUCrop(),
                ResizeWithRandomInterpolation(
                    target_size, max_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    if aug_type == "ssdlite":
        return v2.Compose(  # type: ignore
            [
                v2.ToImage(),
                v2.RandomIoUCrop(),
                ResizeWithRandomInterpolation(
                    target_size, max_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    if aug_type == "detr":
        return v2.Compose(  # type: ignore
            [
                v2.ToImage(),
                v2.RandomChoice(
                    [
                        v2.RandomShortestSize(
                            (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=max_size or 1333
                        ),
                        v2.Compose(
                            [
                                v2.RandomShortestSize((400, 500, 600)),
                                v2.RandomIoUCrop(),  # Supposed to be RandomSizeCrop
                                v2.RandomShortestSize(
                                    (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=max_size or 1333
                                ),
                            ]
                        ),
                    ]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    raise ValueError("Unsupported augmentation type")


def inference_preset(
    size: tuple[int, int], rgv_values: RGBType, dynamic_size: bool = False, max_size: Optional[int] = None
) -> Callable[..., torch.Tensor]:
    """
    Create a torchvision transform pipeline for detection inference

    This function builds a standardized preprocessing pipeline that converts input images
    to tensors with proper normalization and resizing for detection model inference.
    The pipeline handles various sizing strategies.

    Parameters
    ----------
    size
        Target image dimensions as (height, width). Behavior depends on other parameters:
        - With dynamic_size=False and max_size=None: Images resized exactly to this size
        - With dynamic_size=True: min(size) used as target for shorter edge, aspect ratio preserved
        - With max_size specified: Ignored in favor of max_size-based scaling
    rgv_values
        RGB normalization statistics containing 'mean' and 'std' tuples.
        Typically obtained from get_rgb_stats().
    dynamic_size
        When True, preserves aspect ratios by using min(size) as the target
        for the shorter edge. Longer edge scales proportionally.
        Respects max_size is specified.
    max_size
        Maximum allowed size for the longer edge.

    Returns
    -------
    Callable[..., torch.Tensor]
        A callable transform pipeline that takes PIL Images or tensors and returns
        normalized float32 tensors ready for model inference.
    """

    mean = rgv_values["mean"]
    std = rgv_values["std"]
    if dynamic_size is True:
        target_size: Optional[int] | tuple[int, int] = min(size)
    elif max_size is not None:
        target_size = None
    else:
        target_size = size

    return v2.Compose(  # type: ignore
        [
            v2.ToImage(),
            v2.Resize(target_size, interpolation=v2.InterpolationMode.BICUBIC, max_size=max_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor(),
        ]
    )
