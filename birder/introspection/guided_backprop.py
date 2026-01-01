"""
Guided Backpropagation, adapted from
https://github.com/jacobgil/pytorch-grad-cam

Paper "Striving for Simplicity: The All Convolutional Net", https://arxiv.org/abs/1412.6806
"""

# Reference license: MIT

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Function

from birder.introspection.base import InterpretabilityResult
from birder.introspection.base import deprocess_image
from birder.introspection.base import predict_class
from birder.introspection.base import preprocess_image
from birder.introspection.base import validate_target_class


# pylint: disable=abstract-method,arguments-differ
class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx: Any, input_img: torch.Tensor) -> torch.Tensor:
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        ctx.save_for_backward(input_img, output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (input_img, _output) = ctx.saved_tensors

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output, positive_mask_1),
            positive_mask_2,
        )

        return grad_input


# pylint: disable=abstract-method,arguments-differ
class GuidedBackpropSiLU(Function):
    @staticmethod
    def forward(ctx: Any, input_img: torch.Tensor) -> torch.Tensor:
        result = input_img * torch.sigmoid(input_img)
        ctx.save_for_backward(input_img)

        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        positive_mask_1 = (i > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i))) * positive_mask_1 * positive_mask_2

        return grad_input


# pylint: disable=abstract-method,arguments-differ
class GuidedBackpropGELU(Function):
    @staticmethod
    def forward(ctx: Any, input_img: torch.Tensor) -> torch.Tensor:
        result = F.gelu(input_img, approximate="none")  # pylint:disable=not-callable
        ctx.save_for_backward(input_img)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]

        sqrt_2 = math.sqrt(2.0)
        sqrt_2pi = math.sqrt(2.0 * math.pi)

        cdf = 0.5 * (1.0 + torch.erf(x / sqrt_2))
        pdf = torch.exp(-0.5 * x * x) / sqrt_2pi

        d_gelu = cdf + x * pdf

        positive_mask_1 = (x > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)

        grad_input = grad_output * d_gelu * positive_mask_1 * positive_mask_2

        return grad_input


# pylint: disable=abstract-method,arguments-differ
class GuidedBackpropHardswish(Function):
    @staticmethod
    def forward(ctx: Any, input_img: torch.Tensor) -> torch.Tensor:
        result = F.hardswish(input_img)
        ctx.save_for_backward(input_img)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]

        grad = torch.zeros_like(x)

        mask_mid = (x > -3) & (x < 3)
        grad[mask_mid] = (2.0 * x[mask_mid] + 3.0) / 6.0

        mask_high = x >= 3
        grad[mask_high] = 1.0

        positive_mask_1 = (x > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)

        grad_input = grad_output * grad * positive_mask_1 * positive_mask_2

        return grad_input


class GuidedReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GuidedBackpropReLU.apply(x)


class GuidedSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GuidedBackpropSiLU.apply(x)


class GuidedGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GuidedBackpropGELU.apply(x)


class GuidedHardswish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GuidedBackpropHardswish.apply(x)


# Activation replacement mapping
ACTIVATION_REPLACEMENTS: dict[type, type] = {
    nn.ReLU: GuidedReLU,
    nn.SiLU: GuidedSiLU,
    nn.GELU: GuidedGELU,
    nn.Hardswish: GuidedHardswish,
}


def replace_activations_recursive(model: nn.Module, replacements: dict[type, type]) -> None:
    """
    NOTE: This ONLY works for activations defined as nn.Module objects (e.g., self.act = nn.ReLU()).
    It will NOT affect functional calls inside forward methods, such as F.relu(x) or F.gelu(x).
    """

    for name, module in list(model._modules.items()):  # pylint: disable=protected-access
        for old_type, new_type in replacements.items():
            if isinstance(module, old_type):
                model._modules[name] = new_type()  # pylint: disable=protected-access
                break
        else:
            # Recurse into submodules
            replace_activations_recursive(module, replacements)


def restore_activations_recursive(model: nn.Module, guided_types: dict[type, type]) -> None:
    reverse_mapping = {v: k for k, v in guided_types.items()}
    for name, module in list(model._modules.items()):  # pylint: disable=protected-access
        for guided_type, original_type in reverse_mapping.items():
            if isinstance(module, guided_type):
                model._modules[name] = original_type()  # pylint: disable=protected-access
                break
        else:
            restore_activations_recursive(module, guided_types)


class GuidedBackprop:
    def __init__(self, net: nn.Module, device: torch.device, transform: Callable[..., torch.Tensor]) -> None:
        self.net = net.eval()
        self.device = device
        self.transform = transform

    def __call__(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = preprocess_image(image, self.transform, self.device)

        # Get prediction
        with torch.inference_mode():
            logits = self.net(input_tensor)

        if target_class is None:
            target_class = predict_class(logits)
        else:
            validate_target_class(target_class, logits.shape[-1])

        # Replace activations with guided versions
        replace_activations_recursive(self.net, ACTIVATION_REPLACEMENTS)

        try:
            input_tensor = input_tensor.detach().requires_grad_(True)
            output = self.net(input_tensor)

            loss = output[0, target_class]
            loss.backward(retain_graph=False)

            gradients = input_tensor.grad.cpu().numpy()
            gradients = gradients[0, :, :, :]  # Remove batch dim
            gradients = gradients.transpose((1, 2, 0))  # CHW -> HWC

        finally:
            restore_activations_recursive(self.net, ACTIVATION_REPLACEMENTS)

        visualization = deprocess_image(gradients * rgb_img)

        return InterpretabilityResult(
            original_image=rgb_img,
            visualization=visualization,
            raw_output=gradients,
            logits=logits.detach(),
            predicted_class=target_class,
        )
