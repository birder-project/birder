import random
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypedDict

import torch
from torch import nn
from torchvision.transforms import v2

RGBType = TypedDict("RGBType", {"mean": tuple[float, float, float], "std": tuple[float, float, float]})
RGBMode = Literal["birder", "imagenet", "neutral", "none"]


def get_rgb_stats(mode: RGBMode) -> RGBType:
    if mode == "birder":
        return {
            "mean": (0.5248, 0.5372, 0.5086),
            "std": (0.2135, 0.2103, 0.2622),
        }

    if mode == "imagenet":
        return {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

    if mode == "neutral":
        return {
            "mean": (0.0, 0.0, 0.0),
            "std": (1.0, 1.0, 1.0),
        }

    if mode == "none":
        return {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
        }

    raise ValueError(f"unknown mode={mode}")


def get_mixup_cutmix(alpha: Optional[float], num_outputs: int, cutmix: bool) -> Callable[..., torch.Tensor]:
    choices: list[Callable[..., torch.Tensor]] = []
    choices.append(v2.Identity())
    if alpha is not None:
        choices.append(v2.MixUp(alpha=alpha, num_classes=num_outputs))

    if cutmix is True:
        choices.append(v2.CutMix(alpha=1.0, num_classes=num_outputs))

    return v2.RandomChoice(choices)  # type: ignore


# Using transforms v2 mixup, keeping this implementation only as a reference
class RandomMixup(nn.Module):
    """
    Randomly apply Mixup to the provided batch and targets.

    The class implements the data augmentations as described in the paper
    "mixup: Beyond Empirical Risk Minimization"
    https://arxiv.org/abs/1710.09412

    Parameters
    ----------
    num_classes
        Number of classes used for one-hot encoding.
    p
        Probability of the batch being transformed
    alpha
        Hyperparameter of the Beta distribution used for mixup.
    """

    def __init__(self, num_classes: int, p: float, alpha: float) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param must be positive")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha

    def forward(self, batch: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch
            Float tensor of size (B, C, H, W)
        target
            Integer tensor of size (B, )

        Returns
        -------
        Randomly transformed batch.

        Raises
        ------
        ValueError
            On wrong tensor dimensions.
        TypeError
            On bad tensor dtype.
        """

        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")

        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")

        if batch.is_floating_point() is False:
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")

        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        batch = batch.clone()
        target = target.clone()

        if target.ndim == 1:
            # pylint: disable=not-callable
            target = nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return (batch, target)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]  # pylint: disable=protected-access
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return (batch, target)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha})"


class RandomResizedCropWithRandomInterpolation(nn.Module):
    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float],
        ratio: tuple[float, float],
        interpolation: list[v2.InterpolationMode],
    ) -> None:
        super().__init__()
        self.transform = []
        for interp in interpolation:
            self.transform.append(
                v2.RandomResizedCrop(
                    size,
                    scale=scale,
                    ratio=ratio,
                    interpolation=interp,
                    antialias=True,
                )
            )

    def forward(self, x: Any) -> torch.Tensor:
        t = random.choice(self.transform)
        return t(x)


def training_preset(
    size: tuple[int, int], level: int, rgv_values: RGBType, resize_min_scale: Optional[float] = None
) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    if level == -1:  # AutoAugment policy
        if resize_min_scale is None:
            resize_min_scale = 0.08
        return v2.Compose(  # type: ignore
            [
                v2.PILToTensor(),
                RandomResizedCropWithRandomInterpolation(
                    size,
                    scale=(resize_min_scale, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
                ),
                v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET, v2.InterpolationMode.BILINEAR),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 0:
        return v2.Compose(  # type: ignore
            [
                v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 1:
        if resize_min_scale is None:
            resize_min_scale = 0.7
        return v2.Compose(  # type: ignore
            [
                v2.PILToTensor(),
                RandomResizedCropWithRandomInterpolation(
                    size,
                    scale=(resize_min_scale, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
                ),
                v2.RandomRotation(5, fill=0),
                v2.ColorJitter(brightness=0.2, contrast=0.1, hue=0),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 2:
        if resize_min_scale is None:
            resize_min_scale = 0.65
        return v2.Compose(  # type: ignore
            [
                v2.PILToTensor(),
                RandomResizedCropWithRandomInterpolation(
                    size,
                    scale=(resize_min_scale, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
                ),
                v2.RandomChoice(
                    [
                        v2.RandomRotation(10, fill=0),
                        v2.RandomAffine(degrees=0, translate=(0, 0), shear=(-15, 15, 0, 0), fill=0),
                    ]
                ),
                v2.RandomPosterize(7, p=0.25),
                v2.RandomChoice(
                    [
                        v2.RandomAutocontrast(0.5),
                        v2.ColorJitter(brightness=0.225, contrast=0.15, hue=0.02),
                    ]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 3:
        if resize_min_scale is None:
            resize_min_scale = 0.6
        return v2.Compose(  # type: ignore
            [
                v2.PILToTensor(),
                RandomResizedCropWithRandomInterpolation(
                    size,
                    scale=(resize_min_scale, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
                ),
                v2.RandomChoice(
                    [
                        v2.RandomRotation(12, fill=0),
                        v2.RandomAffine(degrees=0, translate=(0, 0), shear=(-20, 20, 0, 0), fill=0),
                    ]
                ),
                v2.RandomPosterize(6, p=0.2),
                v2.RandomChoice(
                    [
                        v2.RandomAutocontrast(0.5),
                        v2.ColorJitter(brightness=0.25, contrast=0.15, hue=0.04),
                    ]
                ),
                v2.RandomChoice(
                    [
                        v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.2))], p=0.5),
                        v2.RandomAdjustSharpness(1.25, p=0.5),
                        v2.RandomAdjustSharpness(1.5, p=0.5),
                    ]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if level == 4:
        if resize_min_scale is None:
            resize_min_scale = 0.5
        return v2.Compose(  # type: ignore
            [
                v2.PILToTensor(),
                RandomResizedCropWithRandomInterpolation(
                    size,
                    scale=(resize_min_scale, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
                ),
                v2.RandomChoice(
                    [
                        v2.RandomRotation(16, fill=0),
                        v2.RandomAffine(degrees=0, translate=(0, 0), shear=(-22, 22, 0, 0), fill=0),
                    ]
                ),
                v2.RandomPosterize(5, p=0.25),
                v2.RandomChoice(
                    [
                        # v2.RandomEqualize(0.25),
                        v2.RandomAutocontrast(0.5),
                        v2.ColorJitter(brightness=0.28, contrast=0.2, hue=0.07),
                    ]
                ),
                v2.RandomChoice(
                    [
                        v2.RandomApply([v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.8, 1.5))], p=0.5),
                        v2.RandomAdjustSharpness(1.5, p=0.5),
                        v2.RandomAdjustSharpness(2.0, p=0.5),
                    ]
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3), value=0),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    raise ValueError("Unsupported level")


def inference_preset(
    size: tuple[int, int], rgv_values: RGBType, center_crop: float = 1.0
) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    base_size = (int(size[0] / center_crop), int(size[1] / center_crop))
    return v2.Compose(  # type: ignore
        [
            v2.Resize(base_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(size),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def reverse_preset(rgv_values: RGBType) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    reverse_mean = [-m / s for m, s in zip(mean, std)]
    reverse_std = [1 / s for s in std]

    return v2.Compose(  # type: ignore
        [
            v2.Normalize(mean=reverse_mean, std=reverse_std),
            v2.ToDtype(torch.uint8, scale=True),
        ]
    )
