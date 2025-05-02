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


class SimpleRandomCropWithRandomInterpolation(nn.Module):
    def __init__(self, size: tuple[int, int], interpolation: list[v2.InterpolationMode]) -> None:
        super().__init__()
        self.transform = []
        for interp in interpolation:
            self.transform.append(
                v2.Compose(
                    [
                        v2.Resize(min(size), interpolation=interp),
                        v2.RandomCrop(size, padding=4, padding_mode="reflect"),
                    ]
                )
            )

    def forward(self, x: Any) -> torch.Tensor:
        t = random.choice(self.transform)
        return t(x)


def birder_augment(level: int, solarize_prob: float = 0.0, grayscale_prob: float = 0.0) -> Callable[..., torch.Tensor]:
    if level == 0:
        return v2.Identity()  # type: ignore

    transforms = []
    if level == 1:
        transforms.extend(
            [
                v2.RandomRotation(5, fill=0),
                v2.ColorJitter(brightness=0.2, contrast=0.1, hue=0),
            ]
        )

    elif level == 2:
        transforms.extend(
            [
                v2.RandomAffine(degrees=10, translate=None, shear=(-15, 15, 0, 0), fill=0),
                v2.RandomPosterize(7, p=0.2),
                v2.RandomChoice(
                    [
                        v2.RandomAutocontrast(0.5),
                        v2.ColorJitter(brightness=0.225, contrast=0.15, hue=0.02),
                    ]
                ),
            ]
        )

    elif level == 3:
        transforms.extend(
            [
                v2.RandomAffine(degrees=15, translate=None, shear=(-20, 20, 0, 0), fill=0),
                v2.RandomPosterize(6, p=0.25),
                v2.ColorJitter(brightness=0.25, contrast=0.15, hue=0.05),
                v2.RandomChoice(
                    [
                        v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.2))], p=0.5),
                        v2.RandomAdjustSharpness(1.25, p=0.5),
                    ]
                ),
            ]
        )

    elif level == 4:
        transforms.extend(
            [
                v2.RandomAffine(degrees=20, translate=None, shear=(-22, 22, 0, 0), fill=0),
                v2.RandomPosterize(5, p=0.25),
                v2.ColorJitter(brightness=0.3, contrast=0.2, hue=0.1),
                v2.RandomChoice(
                    [
                        v2.RandomApply([v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.8, 1.5))], p=0.5),
                        v2.RandomAdjustSharpness(1.5, p=0.5),
                    ]
                ),
            ]
        )

    else:
        raise ValueError("Unsupported level")

    if solarize_prob > 0 and grayscale_prob > 0:
        transforms.append(
            v2.RandomChoice([v2.RandomSolarize(threshold=128, p=solarize_prob), v2.RandomGrayscale(grayscale_prob)])
        )
    elif solarize_prob > 0:
        transforms.append(v2.RandomSolarize(threshold=128, p=solarize_prob))
    elif grayscale_prob > 0:
        transforms.append(v2.RandomGrayscale(grayscale_prob))

    return v2.Compose(transforms)  # type: ignore[no-any-return]


AugType = Literal["birder", "aa", "ra", "ta_wide", "augmix", "3aug"]


# pylint: disable=too-many-branches
def training_preset(
    size: tuple[int, int],
    aug_type: AugType,
    level: int,
    rgv_values: RGBType,
    resize_min_scale: Optional[float] = None,
    re_prob: Optional[float] = None,
    ra_magnitude: int = 9,
    augmix_severity: int = 3,
    solarize_prob: Optional[float] = None,
    grayscale_prob: Optional[float] = None,
    simple_crop: bool = False,
) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    if aug_type == "birder":
        if solarize_prob is None:
            solarize_prob = 0.0
        if grayscale_prob is None:
            grayscale_prob = 0.0

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
            re_scale = 0.1
            if resize_min_scale is None:
                resize_min_scale = 0.65
            if re_prob is None:
                re_prob = 0.0

        elif level == 2:
            re_scale = 0.15
            if resize_min_scale is None:
                resize_min_scale = 0.6
            if re_prob is None:
                re_prob = 0.0

        elif level == 3:
            re_scale = 0.2
            if resize_min_scale is None:
                resize_min_scale = 0.55
            if re_prob is None:
                re_prob = 0.1

        elif level == 4:
            re_scale = 0.25
            if resize_min_scale is None:
                resize_min_scale = 0.45
            if re_prob is None:
                re_prob = 0.2

        else:
            raise ValueError("Unsupported level")

        if simple_crop is True:
            crop_transform = SimpleRandomCropWithRandomInterpolation(
                size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
            )
        else:
            crop_transform = RandomResizedCropWithRandomInterpolation(
                size,
                scale=(resize_min_scale, 1.0),
                ratio=(3 / 4, 4 / 3),
                interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
            )

        return v2.Compose(  # type:ignore
            [
                v2.PILToTensor(),
                crop_transform,
                birder_augment(level, solarize_prob=solarize_prob, grayscale_prob=grayscale_prob),
                v2.RandomHorizontalFlip(0.5),
                v2.Identity() if re_prob == 0 else v2.RandomErasing(re_prob, scale=(0.02, re_scale), ratio=(0.3, 3)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    if resize_min_scale is None:
        resize_min_scale = 0.08
    if re_prob is None:
        re_prob = 0.0

    transforms = [v2.PILToTensor()]
    if simple_crop is True:
        transforms.append(
            SimpleRandomCropWithRandomInterpolation(
                size, interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC]
            )
        )
    else:
        transforms.append(
            RandomResizedCropWithRandomInterpolation(
                size,
                scale=(resize_min_scale, 1.0),
                ratio=(3 / 4, 4 / 3),
                interpolation=[v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC],
            )
        )

    if aug_type == "aa":  # AutoAugment policy
        transforms.append(v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET, v2.InterpolationMode.BILINEAR))
    elif aug_type == "ra":  # RandAugment policy
        transforms.append(v2.RandAugment(magnitude=ra_magnitude, interpolation=v2.InterpolationMode.BILINEAR))
    elif aug_type == "ta_wide":  # TrivialAugmentWide policy
        transforms.append(v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR))
    elif aug_type == "augmix":
        transforms.append(v2.AugMix(severity=augmix_severity, interpolation=v2.InterpolationMode.BILINEAR))
    elif aug_type == "3aug":
        transforms.append(
            v2.RandomChoice(
                [v2.RandomGrayscale(p=1.0), v2.RandomSolarize(128, p=1.0), v2.GaussianBlur(kernel_size=(3, 3))]
            )
        )
        transforms.append(v2.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3))
    else:
        raise ValueError("Unsupported augmentation type")

    return v2.Compose(  # type:ignore
        [
            *transforms,
            v2.RandomHorizontalFlip(0.5),
            v2.Identity() if re_prob == 0 else v2.RandomErasing(re_prob),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


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
