"""
Projected Gradient Descent, adapted from
https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py

Paper "Towards Deep Learning Models Resistant to Adversarial Attacks",
https://arxiv.org/abs/1706.06083
"""

# Reference license: MIT

from typing import NamedTuple
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

PGDResponse = NamedTuple("PGDResponse", [("out", torch.Tensor), ("adv_img", torch.Tensor), ("adv_out", torch.Tensor)])


class PGD:
    def __init__(self, net: nn.Module, eps: float, max_delta: float, steps: int, random_start: bool) -> None:
        self.net = net
        self.max_delta = max_delta
        self.eps = eps
        self.steps = steps
        self.random_start = random_start

    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> PGDResponse:
        adv_image = input_tensor.clone().detach()
        out = self.net(input_tensor)
        if target is None:
            target = torch.argmax(out, dim=1)

        if self.random_start is True:
            # Starting at a uniformly random point
            adv_image = adv_image + torch.empty_like(adv_image).uniform_(-self.max_delta, self.max_delta)
            adv_image = torch.clamp(adv_image, min=-4, max=4).detach()

        for _ in range(self.steps):
            adv_image.requires_grad = True
            outputs = self.net(adv_image)
            loss = F.nll_loss(outputs, target)
            self.net.zero_grad()
            loss.backward()

            grad = adv_image.grad.data
            adv_image = adv_image.detach() + self.eps * grad.sign()
            delta = torch.clamp(adv_image - input_tensor, min=-self.max_delta, max=self.max_delta)
            adv_image = torch.clamp(input_tensor + delta, min=-4, max=4).detach()

        adv_out = self.net(adv_image)

        return PGDResponse(F.softmax(out, dim=1), adv_image, F.softmax(adv_out, dim=1))
