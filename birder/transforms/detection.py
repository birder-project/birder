import math
import random
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torchvision.transforms import v2

from birder.transforms.classification import RGBType


def batch_images(images: list[torch.Tensor], size_divisible: int = 32) -> torch.Tensor:
    """
    Batch list of image tensors of different sizes into a single batch.
    Pad with zeros all images to the shape of the largest image in the list.
    """

    size_list = [list(img.shape) for img in images]
    max_size = size_list[0]
    for sublist in size_list[1:]:
        for index, item in enumerate(sublist):
            max_size[index] = max(max_size[index], item)

    stride = float(size_divisible)
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    batch_shape = [len(images)] + max_size
    batched_imgs = images[0].new_full(batch_shape, 0)
    for i in range(batched_imgs.shape[0]):
        img = images[i]
        batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


class ResizeWithRandomInterpolation(nn.Module):
    def __init__(self, size: tuple[int, int], interpolation: list[v2.InterpolationMode]) -> None:
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


def training_preset(size: tuple[int, int], level: int, rgv_values: RGBType) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]
    fill_value = [255 * v for v in mean]

    if level == 0:
        return v2.Compose(  # type: ignore
            [
                v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                v2.SanitizeBoundingBoxes(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 1:
        return v2.Compose(  # type: ignore
            [
                v2.ScaleJitter(
                    target_size=size,
                    scale_range=(0.25, 2),
                    antialias=True,
                ),
                v2.RandomZoomOut(fill_value),
                ResizeWithRandomInterpolation(
                    size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.SanitizeBoundingBoxes(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 2:
        return v2.Compose(  # type: ignore
            [
                v2.ScaleJitter(
                    target_size=size,
                    scale_range=(0.2, 2),
                    antialias=True,
                ),
                v2.RandomZoomOut(fill_value),
                ResizeWithRandomInterpolation(
                    size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.ColorJitter(brightness=0.25, contrast=0.15, hue=0.04),
                v2.SanitizeBoundingBoxes(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 3:
        return v2.Compose(  # type: ignore
            [
                v2.ScaleJitter(
                    target_size=size,
                    scale_range=(0.1, 2),
                    antialias=True,
                ),
                v2.RandomZoomOut(fill_value),
                v2.RandomIoUCrop(min_scale=0.7),
                ResizeWithRandomInterpolation(
                    size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomAutocontrast(0.2),
                v2.ColorJitter(brightness=0.27, contrast=0.16, hue=0.06),
                v2.SanitizeBoundingBoxes(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    raise ValueError("Unsupported level")


def inference_preset(size: tuple[int, int], rgv_values: RGBType) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    return v2.Compose(  # type: ignore
        [
            v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
