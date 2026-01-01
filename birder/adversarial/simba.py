"""
SimBA (Simple Black-box Attack)

Paper "Simple Black-box Adversarial Attacks", https://arxiv.org/abs/1905.07121
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


class SimBA:
    def __init__(self, net: nn.Module, step_size: float, max_iter: int = 1000, *, rgb_stats: RGBType) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.net = net.eval()
        self.step_size = step_size
        self.max_iter = max_iter
        self.rgb_stats = rgb_stats

    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> AttackResult:
        inputs = input_tensor.detach()
        with torch.no_grad():
            logits = self.net(inputs)

        labels = predict_labels(logits)
        target_labels = (
            validate_target(target, inputs.shape[0], logits.shape[1], inputs.device) if target is not None else None
        )
        targeted = target_labels is not None

        adv_inputs_list = []
        total_queries = 0
        for idx in range(inputs.size(0)):
            label = labels[idx : idx + 1]
            target_label = target_labels[idx : idx + 1] if target_labels is not None else None
            adv_input, num_queries = self._attack_single(inputs[idx : idx + 1], label, target_label)
            adv_inputs_list.append(adv_input)
            total_queries += num_queries

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
            num_queries=total_queries,
        )

    # pylint: disable=too-many-locals
    def _attack_single(
        self, inputs: torch.Tensor, label: torch.Tensor, target_label: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        adv_inputs = inputs.clone()
        num_queries = 1  # Baseline forward pass

        with torch.no_grad():
            current_logits = self.net(adv_inputs)
            current_objective = self._compute_objective(current_logits, label, target_label)

        if self._is_successful(current_logits, label, target_label):
            return adv_inputs.detach(), num_queries

        (_, channels, height, width) = adv_inputs.shape
        num_dims = channels * height * width
        step = pixel_eps_to_normalized(self.step_size, self.rgb_stats, device=adv_inputs.device, dtype=adv_inputs.dtype)
        step_vals = step.view(-1)  # Per-channel steps
        stride = height * width

        perm = torch.randperm(num_dims, device=adv_inputs.device)
        num_steps = min(self.max_iter, num_dims)

        # Coordinate-wise search in random order
        for flat_idx in perm[:num_steps]:
            (c, rem) = divmod(int(flat_idx.item()), stride)
            (h, w) = divmod(rem, width)
            step_val = step_vals[c]

            (candidate_inputs, candidate_logits, candidate_objective) = self._best_candidate(
                adv_inputs, c, h, w, step_val, label, target_label
            )
            num_queries += 2

            if candidate_objective < current_objective:
                adv_inputs = candidate_inputs
                current_logits = candidate_logits
                current_objective = candidate_objective

                if self._is_successful(current_logits, label, target_label) is True:
                    break

        return adv_inputs.detach(), num_queries

    def _perturb_pixel(
        self, inputs: torch.Tensor, channel: int, row: int, col: int, step: torch.Tensor
    ) -> torch.Tensor:
        adv_inputs = inputs.clone()
        adv_inputs[0, channel, row, col] = adv_inputs[0, channel, row, col] + step
        return clamp_normalized(adv_inputs, self.rgb_stats)

    def _evaluate_candidate(
        self, inputs: torch.Tensor, label: torch.Tensor, target_label: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, float]:
        with torch.no_grad():
            logits = self.net(inputs)

        return logits, self._compute_objective(logits, label, target_label)

    def _best_candidate(
        self,
        inputs: torch.Tensor,
        channel: int,
        row: int,
        col: int,
        step: torch.Tensor,
        label: torch.Tensor,
        target_label: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        adv_plus = self._perturb_pixel(inputs, channel, row, col, step)
        logits_plus, objective_plus = self._evaluate_candidate(adv_plus, label, target_label)

        adv_minus = self._perturb_pixel(inputs, channel, row, col, -step)
        logits_minus, objective_minus = self._evaluate_candidate(adv_minus, label, target_label)

        if objective_plus <= objective_minus:
            return adv_plus, logits_plus, objective_plus

        return adv_minus, logits_minus, objective_minus

    @staticmethod
    def _compute_objective(
        logits: torch.Tensor, original_label: torch.Tensor, target_label: Optional[torch.Tensor]
    ) -> float:
        # Lower objective is better in both modes
        if target_label is not None:
            return float(F.cross_entropy(logits, target_label).item())

        return -float(F.cross_entropy(logits, original_label).item())

    @staticmethod
    def _is_successful(
        logits: torch.Tensor, original_label: torch.Tensor, target_label: Optional[torch.Tensor]
    ) -> bool:
        pred = predict_labels(logits)
        if target_label is not None:
            return bool(pred.eq(target_label).item())

        return bool(pred.ne(original_label).item())
