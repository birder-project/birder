"""
Adapted from https://github.com/jacobgil/pytorch-grad-cam

Paper "Striving for Simplicity: The All Convolutional Net", https://arxiv.org/abs/1412.6806
"""

# Reference license: MIT

from typing import Any
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import Function


# pylint: disable=protected-access
def replace_all_layer_type_recursive(model: nn.Module, old_layer_type: nn.Module, new_layer: nn.Module) -> None:
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type) is True:
            model._modules[name] = new_layer

        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)


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


class GuidedBackpropReLUModel:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        return self.model(input_img)

    def __call__(self, input_img: torch.Tensor, target_category: Optional[int] = None) -> torch.Tensor:
        replace_all_layer_type_recursive(self.model, nn.ReLU, GuidedBackpropReLUAsModule())
        replace_all_layer_type_recursive(self.model, nn.GELU, GuidedBackpropGeLUAsModule())
        replace_all_layer_type_recursive(self.model, nn.SiLU, GuidedBackpropSwishAsModule())

        input_img = input_img.requires_grad_(True)
        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy()).item()

        loss = output[0, target_category]
        loss.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))

        replace_all_layer_type_recursive(self.model, GuidedBackpropSwishAsModule, nn.SiLU())
        replace_all_layer_type_recursive(self.model, GuidedBackpropGeLUAsModule, nn.GELU())
        replace_all_layer_type_recursive(self.model, GuidedBackpropReLUAsModule, nn.ReLU())

        return output
