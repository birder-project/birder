"""
Fast Gradient Sign Method (FGSM)

Paper "Explaining and Harnessing Adversarial Examples", https://arxiv.org/abs/1412.6572
"""

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


class FGSM:
    def __init__(self, net: nn.Module, eps: float, *, rgb_stats: RGBType) -> None:
        self.net = net.eval()
        self.eps = eps
        self.rgb_stats = rgb_stats

    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> AttackResult:
        inputs = input_tensor.detach().clone()
        inputs.requires_grad_(True)

        logits = self.net(inputs)
        targeted = target is not None
        if targeted is True:
            target = validate_target(target, inputs.shape[0], logits.shape[1], inputs.device)
        else:
            target = predict_labels(logits)

        loss = F.cross_entropy(logits, target)
        (grad,) = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
        eps_norm = pixel_eps_to_normalized(self.eps, self.rgb_stats, device=inputs.device, dtype=inputs.dtype)

        # Targeted steps descend toward target, untargeted ascend away from original
        if targeted is True:
            direction = -1.0
        else:
            direction = 1.0

        perturbation = direction * eps_norm * grad.sign()
        adv_inputs = clamp_normalized(inputs + perturbation, self.rgb_stats)
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
