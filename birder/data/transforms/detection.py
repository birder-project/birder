import random
from collections.abc import Callable
from typing import Any
from typing import Literal

import torch
from torch import nn
from torchvision.transforms import v2

from birder.data.transforms.classification import RGBType


class ResizeWithRandomInterpolation(nn.Module):
    def __init__(self, size: int | tuple[int, int], interpolation: list[v2.InterpolationMode]) -> None:
        super().__init__()
        self.transform = []
        for interp in interpolation:
            self.transform.append(
                v2.Resize(
                    size,
                    interpolation=interp,
                    antialias=True,
                )
            )

    def forward(self, *x: Any) -> torch.Tensor:
        t = random.choice(self.transform)
        return t(x)


def get_birder_augment(
    size: tuple[int, int], level: int, fill_value: list[float], dynamic_size: bool, multiscale: bool
) -> Callable[..., torch.Tensor]:
    if dynamic_size is True:
        target_size: int | tuple[int, int] = min(size)
    else:
        target_size = size

    transformations = []
    transformations.extend(
        [
            v2.RandomChoice(
                [
                    v2.ScaleJitter(target_size=size, scale_range=(max(0.1, 0.5 - (0.08 * level)), 2), antialias=True),
                    v2.RandomZoomOut(fill_value, side_range=(1, 3 + level * 0.1), p=0.5),
                ]
            ),
        ]
    )

    if level >= 3:
        transformations.extend([v2.RandomIoUCrop()])

    if multiscale is True:
        transformations.append(
            v2.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
        )
    else:
        transformations.append(
            ResizeWithRandomInterpolation(
                target_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
            ),
        )

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


AugType = Literal["birder", "multiscale", "ssd", "ssdlite"]


def training_preset(
    size: tuple[int, int], aug_type: AugType, level: int, rgv_values: RGBType, dynamic_size: bool, multiscale: bool
) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]
    fill_value = [255 * v for v in mean]
    if dynamic_size is True:
        target_size: int | tuple[int, int] = min(size)
    else:
        target_size = size

    if aug_type == "birder":
        if 0 > level or level > 10:
            raise ValueError("Unsupported aug level")

        if level == 0:
            return v2.Compose(  # type: ignore
                [
                    v2.ToImage(),
                    v2.Resize(target_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std),
                    v2.ToPureTensor(),
                ]
            )

        return v2.Compose(  # type:ignore
            [
                v2.ToImage(),
                get_birder_augment(size, level, fill_value, dynamic_size, multiscale),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    if aug_type == "multiscale":
        return v2.Compose(  # type: ignore
            [
                v2.ToImage(),
                v2.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
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
                    target_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
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
                    target_size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToPureTensor(),
            ]
        )

    raise ValueError("Unsupported augmentation type")


def inference_preset(size: tuple[int, int], rgv_values: RGBType, dynamic_size: bool) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]
    if dynamic_size is True:
        target_size: int | tuple[int, int] = min(size)
    else:
        target_size = size

    return v2.Compose(  # type: ignore
        [
            v2.ToImage(),
            v2.Resize(target_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor(),
        ]
    )
