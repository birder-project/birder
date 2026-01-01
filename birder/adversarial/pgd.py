"""
Projected Gradient Descent (PGD)

Paper "Towards Deep Learning Models Resistant to Adversarial Attacks", https://arxiv.org/abs/1706.06083
"""

# Reference license: MIT

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.adversarial.base import AttackResult
from birder.adversarial.base import attack_success
from birder.adversarial.base import clamp_normalized
from birder.adversarial.base import pixel_eps_to_normalized
from birder.adversarial.base import predict_labels
from birder.adversarial.base import validate_target
from birder.data.transforms.classification import RGBType


class PGD:
    def __init__(
        self,
        net: nn.Module,
        eps: float,
        steps: int = 10,
        step_size: Optional[float] = None,
        random_start: bool = False,
        *,
        rgb_stats: RGBType,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be a positive integer")

        self.net = net.eval()
        self.eps = eps
        self.steps = steps
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = eps / steps

        self.random_start = random_start
        self.rgb_stats = rgb_stats

        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> AttackResult:
        inputs = input_tensor.detach()
        with torch.no_grad():
            logits = self.net(inputs)

        targeted = target is not None
        if targeted:
            target = validate_target(target, inputs.shape[0], logits.shape[1], inputs.device)
        else:
            target = predict_labels(logits)

        eps_norm = pixel_eps_to_normalized(self.eps, self.rgb_stats, device=inputs.device, dtype=inputs.dtype)
        step_norm = pixel_eps_to_normalized(self.step_size, self.rgb_stats, device=inputs.device, dtype=inputs.dtype)

        # Targeted steps descend toward target, untargeted ascend away from original
        if targeted is True:
            direction = -1.0
        else:
            direction = 1.0

        adv_inputs = inputs.clone()
        if self.random_start is True:
            # Random start inside the epsilon ball
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs).uniform_(-1.0, 1.0) * eps_norm
            adv_inputs = clamp_normalized(adv_inputs, self.rgb_stats)

        for _ in range(self.steps):
            adv_inputs.requires_grad_(True)
            adv_logits = self.net(adv_inputs)
            loss = F.cross_entropy(adv_logits, target)
            (grad,) = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)
            adv_inputs = adv_inputs.detach() + direction * step_norm * grad.sign()

            # Project back into the epsilon ball around the original input.
            delta = torch.clamp(adv_inputs - inputs, min=-eps_norm, max=eps_norm)
            adv_inputs = clamp_normalized(inputs + delta, self.rgb_stats)

        with torch.no_grad():
            adv_logits = self.net(adv_inputs)

        success = attack_success(
            logits.detach(),
            adv_logits,
            targeted,
            target=target if targeted else None,
        )

        return AttackResult(
            adv_inputs=adv_inputs,
            adv_logits=adv_logits,
            perturbation=adv_inputs - inputs,
            logits=logits.detach(),
            success=success,
        )
