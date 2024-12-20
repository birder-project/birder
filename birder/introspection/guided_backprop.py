"""
Adapted from https://github.com/jacobgil/pytorch-grad-cam

Paper "Striving for Simplicity: The All Convolutional Net", https://arxiv.org/abs/1412.6806
"""

# Reference license: MIT

from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import nn
from torch.autograd import Function

from birder.introspection.base import InterpretabilityResult
from birder.introspection.base import Interpreter


def _deprocess_image(img: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """
    See https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """

    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)

    return np.array(img * 255).astype(np.uint8)


# pylint: disable=protected-access
def _replace_all_layer_type_recursive(model: nn.Module, old_layer_type: nn.Module, new_layer: nn.Module) -> None:
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer

        _replace_all_layer_type_recursive(layer, old_layer_type, new_layer)


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
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output, positive_mask_1),
            positive_mask_2,
        )

        return grad_input


# pylint: disable=abstract-method,arguments-differ
class GuidedBackpropSwish(Function):
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


class GuidedBackpropReLUAsModule(nn.Module):
    def forward(self, input_img: torch.Tensor) -> Any:
        return GuidedBackpropReLU.apply(input_img)


class GuidedBackpropSwishAsModule(nn.Module):
    def forward(self, input_img: torch.Tensor) -> Any:
        return GuidedBackpropSwish.apply(input_img)


class GuidedBackpropGeLUAsModule(nn.Module):
    def forward(self, input_img: torch.Tensor) -> Any:
        return GuidedBackpropSwish.apply(input_img)


class GuidedBackpropHardswishAsModule(nn.Module):
    def forward(self, input_img: torch.Tensor) -> Any:
        return GuidedBackpropSwish.apply(input_img)


class GuidedBackpropModel:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        return self.model(input_img)

    def __call__(self, input_img: torch.Tensor, target_category: Optional[int] = None) -> npt.NDArray[np.float32]:
        _replace_all_layer_type_recursive(self.model, nn.ReLU, GuidedBackpropReLUAsModule())
        _replace_all_layer_type_recursive(self.model, nn.GELU, GuidedBackpropGeLUAsModule())
        _replace_all_layer_type_recursive(self.model, nn.SiLU, GuidedBackpropSwishAsModule())
        _replace_all_layer_type_recursive(self.model, nn.Hardswish, GuidedBackpropHardswishAsModule())

        input_img = input_img.requires_grad_(True)
        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy()).item()

        loss = output[0, target_category]
        loss.backward(retain_graph=True)

        output_grad = input_img.grad.cpu().data.numpy()
        output_grad = output_grad[0, :, :, :]
        output_grad = output_grad.transpose((1, 2, 0))

        _replace_all_layer_type_recursive(self.model, GuidedBackpropHardswishAsModule, nn.Hardswish())
        _replace_all_layer_type_recursive(self.model, GuidedBackpropSwishAsModule, nn.SiLU())
        _replace_all_layer_type_recursive(self.model, GuidedBackpropGeLUAsModule, nn.GELU())
        _replace_all_layer_type_recursive(self.model, GuidedBackpropReLUAsModule, nn.ReLU())

        return output_grad  # type: ignore[no-any-return]


class GuidedBackpropInterpreter(Interpreter):
    def interpret(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = self._preprocess_image(image)

        guided_bp = GuidedBackpropModel(self.model)
        bp_img = guided_bp(input_tensor, target_category=target_class)

        return InterpretabilityResult(rgb_img, _deprocess_image(bp_img * rgb_img), raw_output=bp_img)
