from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Protocol

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


@dataclass(frozen=True)
class InterpretabilityResult:
    original_image: npt.NDArray[np.float32]
    visualization: npt.NDArray[np.float32] | npt.NDArray[np.uint8]
    raw_output: npt.NDArray[np.float32]
    logits: Optional[torch.Tensor] = None
    predicted_class: Optional[int] = None

    def show(self, figsize: tuple[int, int] = (12, 8)) -> None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(self.visualization)
        ax2.imshow(self.original_image)
        plt.show()


class Interpreter(Protocol):
    def __call__(
        self, image: str | Path | Image.Image, target_class: Optional[int] = None
    ) -> InterpretabilityResult: ...


def load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image)

    return image


def preprocess_image(
    image: str | Path | Image.Image, transform: Callable[..., torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, npt.NDArray[np.float32]]:
    pil_image = load_image(image)
    input_tensor = transform(pil_image).unsqueeze(dim=0).to(device)

    # Resize and normalize for visualization
    resized = pil_image.resize((input_tensor.shape[-1], input_tensor.shape[-2]))
    rgb_img = np.array(resized).astype(np.float32) / 255.0

    return (input_tensor, rgb_img)


def show_mask_on_image(
    img: npt.NDArray[np.float32],
    mask: npt.NDArray[np.float32],
    image_weight: float = 0.5,
    colormap: str = "jet",
) -> npt.NDArray[np.uint8]:
    color_map = matplotlib.colormaps[colormap]
    heatmap = color_map(mask)[:, :, :3]

    cam: npt.NDArray[np.float32] = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    cam = cam * 255

    return cam.astype(np.uint8)


def scale_cam_image(
    cam: npt.NDArray[np.float32], target_size: Optional[tuple[int, int]] = None
) -> npt.NDArray[np.float32]:
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = np.array(Image.fromarray(img).resize(target_size))

        result.append(img)

    return np.array(result, dtype=np.float32)


def deprocess_image(img: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)

    return np.array(img * 255).astype(np.uint8)


def validate_target_class(target_class: Optional[int], num_classes: int) -> None:
    if target_class is not None:
        if target_class < 0 or target_class >= num_classes:
            raise ValueError(f"target_class must be in range [0, {num_classes}), got {target_class}")


def predict_class(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())
