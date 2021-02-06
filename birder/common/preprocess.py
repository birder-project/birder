import math
from typing import Tuple

import mxnet as mx
import numpy as np

DEFAULT_RGB = {
    "mean_r": 127.5,
    "mean_g": 127.5,
    "mean_b": 127.5,
    "std_r": 1,
    "std_g": 1,
    "std_b": 1,
    "scale": 1 / 127.5,
}

IMAGENET_RGB = {
    "mean_r": 123.68,
    "mean_g": 116.779,
    "mean_b": 103.939,
    "std_r": 58.393,
    "std_g": 57.12,
    "std_b": 57.375,
    "scale": 1,
}


def preprocess_image(
    img: mx.nd.NDArray,
    size: Tuple[int, int],
    rgb_mean: mx.nd.NDArray,
    rgb_std: mx.nd.NDArray,
    rgb_scale: float,
    center_crop: float = 1.0,
) -> mx.nd.NDArray:
    """
    Preprocess image for inference
    """

    if center_crop < 1.0:
        resize0 = int(math.ceil(img.shape[0] * center_crop))
        resize1 = int(math.ceil(img.shape[1] * center_crop))
        img, (_x, _y, _width, _height) = mx.image.center_crop(img, (resize1, resize0), interp=2)

    img = mx.image.imresize(img, size[0], size[1], interp=2)
    img = img.astype(np.float32)
    img = img - rgb_mean
    img = img / rgb_std
    img = img * rgb_scale

    # Convert to NCHW format, e.g. (1, 3, H, W)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)

    return img


def deprocess_image(
    img: mx.nd.NDArray, rgb_mean: mx.nd.NDArray, rgb_std: mx.nd.NDArray, rgb_scale: float
) -> mx.nd.NDArray:
    """
    Reverse the preprocess
    """

    # Convert to HWC format, e.g. (H, W, 3)
    img = mx.nd.squeeze(img)
    img = img.transpose((1, 2, 0))  # pylint: disable=no-member

    img = img / rgb_scale
    img = img * rgb_std
    img = img + rgb_mean
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    return img
