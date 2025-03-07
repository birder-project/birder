"""
Lamb, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lamb.py

Paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes", https://arxiv.org/abs/1904.00962
"""

# Reference license: Apache-2.0

import math
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import ParamsT


class Lamb(Optimizer):
    """
    Lamb for PyTorch

    Taken as-is from timm - https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lamb.py
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        bias_correction: bool = True,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        grad_averaging: bool = True,
        max_grad_norm: Optional[float] = 1.0,
        trust_clip: bool = False,
        always_adapt: bool = False,
        caution: bool = False,
        decoupled_decay: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        params
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr
            Learning rate.
        bias_correction
            Whether to use bias correction for the moment estimates, similar to Adam.
        betas
            Coefficients used for computing running averages of gradient and its norm.
        eps
            Term added to the denominator to improve numerical stability.
        weight_decay
            Weight decay (L2 penalty).
        grad_averaging
            Whether apply (1-beta2) to grad when calculating running averages of gradient.
        max_grad_norm
            Value used to clip global gradient norm. Set to None to disable.
        trust_clip
            When True, enables clipping of the LAMB trust ratio to prevent extreme values.
        always_adapt
            When True, always applies LAMB learning rate adaptation.
            When False, only applies adaptation when group weight_decay != 0.
        caution
            Apply caution. See "Cautious Optimizers": https://arxiv.org/abs/2411.16085
        decoupled_decay
            When True, applies weight decay separately from the adaptive updates, similar to AdamW vs Adam.
        """

        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "grad_averaging": grad_averaging,
            "max_grad_norm": max_grad_norm,
            "trust_clip": trust_clip,
            "always_adapt": always_adapt,
            "caution": caution,
            "decoupled_decay": decoupled_decay,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("caution", False)
            group.setdefault("decoupled_decay", False)

    def _get_clip_grad_norm(self) -> Optional[torch.Tensor]:
        max_grad_norm = self.defaults["max_grad_norm"]
        if max_grad_norm is None:
            return None

        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instead.")
                norms.append(torch.linalg.vector_norm(grad))  # pylint: disable=not-callable
        global_norm = torch.linalg.vector_norm(torch.stack(norms))  # pylint: disable=not-callable
        clip_global_norm = (global_norm / max_grad_norm).clamp_(min=1.0)
        return clip_global_norm

    # pylint: disable=too-many-branches
    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_grad_norm = self._get_clip_grad_norm()  # None if disabled

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            grad_averaging = 1 if group["grad_averaging"] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1 ** group["step"]
                bias_correction2 = 1 - beta2 ** group["step"]
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if clip_grad_norm is not None:
                    grad.div_(clip_grad_norm)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update = (exp_avg / bias_correction1).div_(denom)

                if group["caution"]:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (update * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    update.mul_(mask)

                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    if group.get("decoupled_decay", False):
                        p.add_(p, alpha=-group["lr"] * weight_decay)
                    else:
                        update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group["always_adapt"]:
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    trust_ratio = w_norm / g_norm
                    # FIX: nested where required since logical and/or not working in PT XLA
                    # Set the ratio to 1.0 (no change) if either weight norm or grad norm is zero
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if group["trust_clip"]:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.clamp(trust_ratio, max=1.0)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group["lr"])

        return loss
