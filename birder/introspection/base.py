from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import nn


def show_mask_on_image(
    img: npt.NDArray[np.float32], mask: npt.NDArray[np.float32], image_weight: float = 0.5
) -> npt.NDArray[np.float32]:
    color_map = matplotlib.colormaps["jet"]
    heatmap = color_map(mask)[:, :, :3]

    cam: npt.NDArray[np.float32] = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    cam = cam * 255

    return cam.astype(np.uint8)


@dataclass
class InterpretabilityResult:
    original_image: npt.NDArray[np.float32]
    visualization: npt.NDArray[np.float32] | npt.NDArray[np.uint8]
    raw_output: npt.NDArray[np.float32]

    def show(self, figsize: tuple[int, int] = (12, 8)) -> None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(self.visualization)
        ax2.imshow(self.original_image)
        plt.show()


class Interpreter:
    def __init__(self, model: nn.Module, device: torch.device, transform: Callable[..., torch.Tensor]) -> None:
        self.model = model.eval()
        self.device = device
        self.transform = transform

    def interpret(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        raise NotImplementedError

    def _preprocess_image(self, image: str | Path | Image.Image) -> tuple[torch.Tensor, npt.NDArray[np.float32]]:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Transform for model
        input_tensor = self.transform(image).unsqueeze(dim=0).to(self.device)

        # Store original for visualization
        rgb_img = np.array(image.resize(input_tensor.shape[-2:])).astype(np.float32) / 255.0

        return (input_tensor, rgb_img)
