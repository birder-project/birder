"""
Gradient-weighted Class Activation Mapping (Grad-CAM), adapted from
https://github.com/jacobgil/pytorch-grad-cam

Paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
https://arxiv.org/abs/1610.02391
"""

# Reference license: MIT

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import nn
from torch.utils.hooks import RemovableHandle

from birder.introspection.base import InterpretabilityResult
from birder.introspection.base import predict_class
from birder.introspection.base import preprocess_image
from birder.introspection.base import scale_cam_image
from birder.introspection.base import show_mask_on_image
from birder.introspection.base import validate_target_class


def compute_cam(activations: npt.NDArray[np.float32], gradients: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    weights: npt.NDArray[np.float32] = np.mean(gradients, axis=(2, 3))
    weighted_activations = weights[:, :, None, None] * activations
    cam: npt.NDArray[np.float32] = weighted_activations.sum(axis=1)
    cam = np.maximum(cam, 0)

    return cam


class ActivationCapture:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: list[RemovableHandle] = []

        # Register hooks
        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        self.handles.append(target_layer.register_forward_hook(self._save_gradient))

    def _save_activation(self, _module: nn.Module, _input: torch.Tensor, output: torch.Tensor) -> None:
        if self.reshape_transform is not None:
            output = self.reshape_transform(output)

        self.activations = output.cpu().detach()

    def _save_gradient(self, _module: nn.Module, _input: torch.Tensor, output: torch.Tensor) -> None:
        if hasattr(output, "requires_grad") is False or output.requires_grad is False:
            return

        def _store_grad(grad: torch.Tensor) -> None:
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)

            self.gradients = grad.cpu().detach()

        output.register_hook(_store_grad)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def release(self) -> None:
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(
        self,
        net: nn.Module,
        device: torch.device,
        transform: Callable[..., torch.Tensor],
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.net = net.eval()
        self.device = device
        self.transform = transform
        self.target_layer = target_layer

        self.activation_capture = ActivationCapture(net, target_layer, reshape_transform)

    def __call__(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = preprocess_image(image, self.transform, self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.activation_capture(input_tensor)

        # Determine target class
        if target_class is None:
            target_class = predict_class(logits)
        else:
            validate_target_class(target_class, logits.shape[-1])

        # Backward pass
        self.net.zero_grad()
        loss = logits[0, target_class]
        loss.backward(retain_graph=False)

        # Get captured activations and gradients
        if self.activation_capture.activations is None:
            raise RuntimeError("No activations captured")

        if self.activation_capture.gradients is None:
            raise RuntimeError("No gradients captured")

        activations = self.activation_capture.activations.numpy()
        gradients = self.activation_capture.gradients.numpy()

        # Compute CAM
        cam = compute_cam(activations, gradients)
        target_size = (input_tensor.size(-1), input_tensor.size(-2))
        cam_scaled = scale_cam_image(cam, target_size)
        grayscale_cam = cam_scaled[0]

        # Create visualization
        visualization = show_mask_on_image(rgb_img, grayscale_cam)

        return InterpretabilityResult(
            original_image=rgb_img,
            visualization=visualization,
            raw_output=grayscale_cam,
            logits=logits.detach(),
            predicted_class=target_class,
        )

    def __del__(self) -> None:
        if hasattr(self, "activation_capture") is True:
            self.activation_capture.release()
