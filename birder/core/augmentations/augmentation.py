"""
Some implementation details taken from https://github.com/bethgelab/imagecorruptions

Also see paper https://arxiv.org/abs/1807.01697
"""

import random

import cv2
import numpy as np
import skimage
from skimage import filters

from birder.core.augmentations.base import Augmentation
from birder.core.augmentations.base import PostAugmentation
from birder.core.augmentations.helpers import clipped_zoom
from birder.core.augmentations.helpers import disk
from birder.core.augmentations.helpers import motion_blur


class Snow(Augmentation):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        factors = (0.05, 0.275, 3, 0.5, 10, 4.0, 0.95)

        img = np.array(img, dtype=np.float32) / 255.0
        snow_layer = np.random.normal(size=img.shape[:2], loc=factors[0], scale=factors[1])

        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], factors[2])
        snow_layer[snow_layer < factors[3]] = 0
        snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

        snow_layer = motion_blur(
            snow_layer, radius=factors[4], sigma=factors[5], angle=np.random.uniform(-135, -45)
        )

        # The snow layer is rounded and cropped to the img dims
        snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.0
        snow_layer = snow_layer[..., np.newaxis]
        snow_layer = snow_layer[: img.shape[0], : img.shape[1], :]

        img = factors[6] * img + (1 - factors[6]) * np.maximum(
            img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(img.shape[0], img.shape[1], 1) * 1.5 + 0.5
        )

        img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

        return np.uint8(img)


class ImpulseNoise(Augmentation):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        factor = 0.005

        img = skimage.util.random_noise(img / 255.0, mode="s&p", amount=factor)
        img = np.clip(img, 0, 1) * 255

        return np.uint8(img)


class DefocusBlur(Augmentation):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        factors = (2, 0.05)

        img = np.array(img) / 255.0
        kernel = disk(radius=factors[0], alias_blur=factors[1])

        channels = []

        for i in range(3):
            channels.append(cv2.filter2D(img[:, :, i], -1, kernel))

        channels = np.array(channels).transpose((1, 2, 0))
        img = np.clip(channels, 0, 1) * 255

        return np.uint8(img)


class Spatter(Augmentation):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        factors = (0.65, 0.3, 4, 0.69, 0.6, 0)
        img = np.array(img, dtype=np.float32) / 255.0

        liquid_layer = np.random.normal(size=img.shape[:2], loc=factors[0], scale=factors[1])
        liquid_layer = filters.gaussian(liquid_layer, sigma=factors[2])
        liquid_layer[liquid_layer < factors[3]] = 0

        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        (_, dist) = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, kernel)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        liquid_layer = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        liquid_layer /= np.max(liquid_layer, axis=(0, 1))
        liquid_layer *= factors[4]

        # Water is pale turquoise
        color = np.concatenate(
            (
                175 / 255.0 * np.ones_like(liquid_layer[..., :1]),
                238 / 255.0 * np.ones_like(liquid_layer[..., :1]),
                238 / 255.0 * np.ones_like(liquid_layer[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img = cv2.cvtColor(np.clip(img + liquid_layer * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255

        return np.uint8(img)


class RandomCrop(PostAugmentation):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        height = int(img.shape[0] * 0.925)
        width = int(img.shape[1] * 0.925)
        x = random.randint(2, img.shape[1] - width)
        y = random.randint(2, img.shape[0] - height)

        sides = ["left", "right", "top", "bottom"]
        side = random.choice(sides)
        if side == "left":
            img = img[:, x:]

        elif side == "right":
            img = img[:, 0 : x + width]

        elif side == "top":
            img = img[y:, :]

        elif side == "bottom":
            img = img[0 : y + height, :]

        return img


class Rotate(PostAugmentation):
    priority = 50

    def __call__(self, img: np.ndarray) -> np.ndarray:
        rotation_degrees = np.random.uniform(-5, 5)

        (height, width) = img.shape[:2]
        center = (width // 2, height // 2)
        scale = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, scale)
        img = cv2.warpAffine(img, rotation_matrix, (width, height))

        return img
