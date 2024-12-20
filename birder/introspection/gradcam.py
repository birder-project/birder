"""
Adapted from https://github.com/jacobgil/pytorch-grad-cam
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
from birder.introspection.base import Interpreter
from birder.introspection.base import show_mask_on_image


def _scale_cam_image(
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


class ClassifierOutputTarget:
    def __init__(self, category: int) -> None:
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if len(model_output.shape) == 1:
            return model_output[self.category]

        return model_output[:, self.category]


class ActivationsAndGradients:
    """
    Class for extracting activations and
    registering gradients from targeted intermediate layers
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        self.model = model
        self.gradients: torch.Tensor
        self.activations: torch.Tensor
        self.reshape_transform = reshape_transform
        self.handles: list[RemovableHandle] = []

        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        # Because of https://github.com/pytorch/pytorch/issues/61519,
        # we don't use backward hook to record gradients.
        self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, _module: nn.Module, _input: torch.Tensor, output: torch.Tensor) -> None:
        if self.reshape_transform is not None:
            output = self.reshape_transform(output)

        self.activations = output.cpu().detach()

    def save_gradient(self, _module: nn.Module, _input: torch.Tensor, output: torch.Tensor) -> None:
        if hasattr(output, "requires_grad") is False or output.requires_grad is False:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
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
        model: nn.Module,
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer, reshape_transform)

    def get_cam_image(
        self, activations: npt.NDArray[np.float32], grads: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        weights: npt.NDArray[np.float32] = np.mean(grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations
        cam: npt.NDArray[np.float32] = weighted_activations.sum(axis=1)

        return cam

    def compute_layer_cam(self, input_tensor: torch.Tensor) -> npt.NDArray[np.float32]:
        target_size = (input_tensor.size(-1), input_tensor.size(-2))

        layer_activations = self.activations_and_grads.activations.numpy()
        layer_grads = self.activations_and_grads.gradients.numpy()

        cam = self.get_cam_image(layer_activations, layer_grads)
        cam = np.maximum(cam, 0)
        scaled = _scale_cam_image(cam, target_size)
        return scaled[:, None, :]

    def __call__(
        self, input_tensor: torch.Tensor, target: Optional[ClassifierOutputTarget] = None
    ) -> npt.NDArray[np.float32]:
        output = self.activations_and_grads(input_tensor)
        if target is None:
            category = np.argmax(output.cpu().data.numpy(), axis=-1)
            target = ClassifierOutputTarget(category)

        self.model.zero_grad()
        loss = target(output)
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_layer_cam(input_tensor)
        cam_per_layer = np.mean(cam_per_layer, axis=1)
        cam_per_layer = _scale_cam_image(cam_per_layer)

        return cam_per_layer

    def __del__(self) -> None:
        self.activations_and_grads.release()


class GradCamInterpreter(Interpreter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transform: Callable[..., torch.Tensor],
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(model, device, transform)
        self.grad_cam = GradCAM(model, target_layer, reshape_transform=reshape_transform)

    def interpret(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = self._preprocess_image(image)

        if target_class is not None:
            target = ClassifierOutputTarget(target_class)
        else:
            target = None

        grayscale_cam = self.grad_cam(input_tensor, target=target)[0, :]
        visualization = show_mask_on_image(rgb_img, grayscale_cam)

        return InterpretabilityResult(rgb_img, visualization, raw_output=grayscale_cam)
