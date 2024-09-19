from typing import NamedTuple
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

FGSMResponse = NamedTuple(
    "FGSMResponse", [("out", torch.Tensor), ("perturbation", torch.Tensor), ("adv_out", torch.Tensor)]
)


class FGSM:
    def __init__(self, net: nn.Module, eps: float) -> None:
        self.net = net.eval()
        self.eps = eps

    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> FGSMResponse:
        input_tensor.requires_grad = True
        out = self.net(input_tensor)
        if target is None:
            target = torch.argmax(out, dim=1)

        loss = F.nll_loss(out, target)
        self.net.zero_grad()
        loss.backward()

        input_grad = input_tensor.grad.data
        sign_data_grad = input_grad.sign()
        perturbed_image = input_tensor + self.eps * sign_data_grad

        adv_out = self.net(perturbed_image)

        return FGSMResponse(F.softmax(out, dim=1), self.eps * sign_data_grad, F.softmax(adv_out, dim=1))
