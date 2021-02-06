import math

import cv2
import numpy as np
from scipy import ndimage


def gauss_function(x: np.ndarray, sigma: float) -> np.ndarray:
    return (np.exp(-(x ** 2) / (2 * (sigma ** 2)))) / (np.sqrt(2 * np.pi) * sigma)


def get_motion_blur_kernel(width: int, sigma: float) -> np.ndarray:
    kernel = gauss_function(np.arange(width), sigma)
    norm = np.sum(kernel)

    return kernel / norm


def clipped_zoom(img: np.ndarray, zoom_factor: int) -> np.ndarray:
    # Clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / zoom_factor))
    top0 = (img.shape[0] - ch0) // 2

    # Clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / zoom_factor))
    top1 = (img.shape[1] - ch1) // 2

    img = ndimage.zoom(img[top0 : top0 + ch0, top1 : top1 + ch1], (zoom_factor, zoom_factor, 1), order=1)

    return img


def shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1 : dx]

    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx : dx + 1]

    else:
        shifted = image

    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1 : dy, :]

    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy : dy + 1, :]

    return shifted


def motion_blur(x: np.ndarray, radius: int, sigma: float, angle: float) -> np.ndarray:
    width = radius * 2 + 1
    kernel = get_motion_blur_kernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            # Simulated motion exceeded image borders
            break

        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted

    return blurred


def disk(radius: int, alias_blur: float = 0.1, dtype: type = np.float32) -> np.ndarray:
    if radius <= 8:
        grid_range = np.arange(-8, 8 + 1)
        ksize = (3, 3)

    else:
        grid_range = np.arange(-radius, radius + 1)
        ksize = (5, 5)

    x, y = np.meshgrid(grid_range, grid_range)
    aliased_disk = np.array((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # Supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
