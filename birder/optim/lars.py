"""
LARS, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lars.py

Paper "Large Batch Training of Convolutional Networks", https://arxiv.org/abs/1708.03888
"""

# Reference license: Apache-2.0

from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import ParamsT


class Lars(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) for PyTorch

    Taken as-is from timm - https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lars.py
    Type annotations added.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        trust_coeff: float = 0.001,
        eps: float = 1e-8,
        trust_clip: bool = False,
        always_adapt: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        params
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr
            Learning rate.
        momentum
            Momentum factor. Enables momentum-based acceleration.
        dampening
            Dampening for momentum. Reduces the contribution of previous gradients.
        weight_decay
            Weight decay (L2 penalty).
        nesterov
            Enables Nesterov momentum.
        trust_coeff
            Trust coefficient for computing adaptive lr / trust_ratio.
            Controls how aggressively to adapt the learning rate.
        eps
            Small constant added to denominator for numerical stability.
        trust_clip
            When True, enables clipping of the LARS trust ratio to prevent extreme values.
        always_adapt
            When True, always applies LARS learning rate adaptation.
            When False, only applies adaptation when group weight_decay != 0.
        """

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "trust_coeff": trust_coeff,
            "eps": eps,
            "trust_clip": trust_clip,
            "always_adapt": always_adapt,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            trust_coeff = group["trust_coeff"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Apply LARS LR adaptation, LARC clipping, weight decay
                # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
                if weight_decay != 0 or group["always_adapt"]:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # FIX: nested where required since logical and/or not working in PT XLA
                    # Set the ratio to 1.0 (no change) if either weight norm or grad norm is zero
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if group["trust_clip"]:
                        trust_ratio = torch.clamp(trust_ratio / group["lr"], max=1.0)
                    grad.add_(p, alpha=weight_decay)
                    grad.mul_(trust_ratio)

                # Apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                p.add_(grad, alpha=-group["lr"])

        return loss
