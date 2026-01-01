"""
DeepFool

Paper "DeepFool: a simple and accurate method to fool deep neural networks", https://arxiv.org/abs/1511.04599
"""

from typing import Optional

import torch
from torch import nn

from birder.adversarial.base import AttackResult
from birder.adversarial.base import attack_success
from birder.adversarial.base import clamp_normalized
from birder.adversarial.base import predict_labels
from birder.adversarial.base import validate_target
from birder.data.transforms.classification import RGBType

GRAD_EPS = 1e-12


class DeepFool:
    def __init__(
        self, net: nn.Module, num_classes: int = 10, overshoot: float = 0.02, max_iter: int = 50, *, rgb_stats: RGBType
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if overshoot < 0:
            raise ValueError("overshoot must be non-negative")

        self.net = net.eval()
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.rgb_stats = rgb_stats

    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> AttackResult:
        inputs = input_tensor.detach()
        with torch.no_grad():
            logits = self.net(inputs)

        target_labels = (
            validate_target(target, inputs.shape[0], logits.shape[1], inputs.device) if target is not None else None
        )
        targeted = target_labels is not None

        adv_inputs_list = []
        for idx in range(inputs.size(0)):
            target_label = target_labels[idx : idx + 1] if target_labels is not None else None
            adv_input = self._attack_single(inputs[idx : idx + 1], logits[idx : idx + 1], target_label)
            adv_inputs_list.append(adv_input)

        adv_inputs = torch.concat(adv_inputs_list, dim=0)
        with torch.no_grad():
            adv_logits = self.net(adv_inputs)

        success = attack_success(
            logits,
            adv_logits,
            targeted,
            target=target_labels if targeted else None,
        )

        return AttackResult(
            adv_inputs=adv_inputs,
            adv_logits=adv_logits,
            perturbation=adv_inputs - inputs,
            logits=logits.detach(),
            success=success,
        )

    def _attack_single(
        self, inputs: torch.Tensor, logits: torch.Tensor, target_label: Optional[torch.Tensor]
    ) -> torch.Tensor:
        adv_inputs = inputs.clone()
        original_label = int(predict_labels(logits).item())
        targeted = target_label is not None
        for _ in range(self.max_iter):
            adv_inputs.requires_grad_(True)
            outputs = self.net(adv_inputs)
            current_label = int(predict_labels(outputs).item())

            if targeted is True:
                assert target_label is not None
                target_value = int(target_label.item())
                if current_label == target_value:
                    break

                perturbation = self._targeted_perturbation(adv_inputs, outputs, current_label, target_value)

            else:
                if current_label != original_label:
                    break

                perturbation = self._untargeted_perturbation(adv_inputs, outputs, current_label)

            if perturbation is None:
                break

            # Overshoot helps ensure boundary crossing
            adv_inputs = adv_inputs.detach() + (1.0 + self.overshoot) * perturbation
            adv_inputs = clamp_normalized(adv_inputs, self.rgb_stats)

        return adv_inputs.detach()

    def _targeted_perturbation(
        self, adv_inputs: torch.Tensor, outputs: torch.Tensor, current_label: int, target_label: int
    ) -> Optional[torch.Tensor]:
        self.net.zero_grad(set_to_none=True)
        grad_current = torch.autograd.grad(outputs[0, current_label], adv_inputs, retain_graph=True)[0]
        grad_target = torch.autograd.grad(outputs[0, target_label], adv_inputs, retain_graph=False)[0]

        # Direction toward the target boundary
        w = grad_target - grad_current
        w_norm = torch.norm(w.view(-1))
        if w_norm.item() < GRAD_EPS:
            return None

        # Distance to the decision boundary
        f = outputs[0, target_label] - outputs[0, current_label]
        perturbation = (f.abs() / (w_norm**2 + GRAD_EPS)) * w

        return perturbation

    def _untargeted_perturbation(
        self, adv_inputs: torch.Tensor, outputs: torch.Tensor, current_label: int
    ) -> Optional[torch.Tensor]:
        # Search the top-k competing classes
        top_k = min(self.num_classes, outputs.shape[1])
        top_indices = torch.topk(outputs, k=top_k, dim=1).indices[0]
        candidate_labels = [int(idx) for idx in top_indices if int(idx) != current_label]

        if len(candidate_labels) == 0:
            return None

        self.net.zero_grad(set_to_none=True)
        grad_current = torch.autograd.grad(outputs[0, current_label], adv_inputs, retain_graph=True)[0]

        # Track the closest decision boundary
        best_dist = None
        best_w = None
        best_f = None
        for idx, label in enumerate(candidate_labels):
            # Keep the graph until the last class
            retain_graph = idx != len(candidate_labels) - 1
            grad_other = torch.autograd.grad(outputs[0, label], adv_inputs, retain_graph=retain_graph)[0]

            w_k = grad_other - grad_current
            w_norm = torch.norm(w_k.view(-1))
            if w_norm.item() < GRAD_EPS:
                continue

            f_k = outputs[0, label] - outputs[0, current_label]
            dist = f_k.abs() / (w_norm + GRAD_EPS)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_w = w_k
                best_f = f_k

        if best_w is None or best_f is None:
            return None

        # Minimal perturbation toward the closest boundary
        best_w_norm = torch.norm(best_w.view(-1))
        if best_w_norm.item() < GRAD_EPS:
            return None

        perturbation = (best_f.abs() / (best_w_norm**2 + GRAD_EPS)) * best_w

        return perturbation
