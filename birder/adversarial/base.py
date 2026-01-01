from dataclasses import dataclass
from typing import Optional
from typing import Protocol

import torch

from birder.data.transforms.classification import RGBType


@dataclass(frozen=True)
class AttackResult:
    adv_inputs: torch.Tensor
    adv_logits: torch.Tensor
    perturbation: torch.Tensor
    logits: Optional[torch.Tensor] = None
    success: Optional[torch.Tensor] = None
    num_queries: Optional[int] = None


class Attack(Protocol):
    def __call__(self, input_tensor: torch.Tensor, target: Optional[torch.Tensor]) -> AttackResult: ...


def _to_channel_tensor(
    values: tuple[float, float, float], device: Optional[torch.device], dtype: Optional[torch.dtype]
) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def normalized_bounds(
    rgb_stats: RGBType, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _to_channel_tensor(rgb_stats["mean"], device=device, dtype=dtype)
    std = _to_channel_tensor(rgb_stats["std"], device=device, dtype=dtype)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std

    return (min_val, max_val)


def pixel_eps_to_normalized(
    eps: float | torch.Tensor,
    rgb_stats: RGBType,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    eps_tensor = torch.as_tensor(eps, device=device, dtype=dtype)
    std = _to_channel_tensor(rgb_stats["std"], device=eps_tensor.device, dtype=eps_tensor.dtype)

    if eps_tensor.numel() == 1:
        eps_tensor = eps_tensor.reshape(1, 1, 1, 1)
    else:
        eps_tensor = eps_tensor.reshape(1, -1, 1, 1)

    return eps_tensor / std


def clamp_normalized(inputs: torch.Tensor, rgb_stats: RGBType) -> torch.Tensor:
    (min_val, max_val) = normalized_bounds(rgb_stats, device=inputs.device, dtype=inputs.dtype)
    return torch.clamp(inputs, min=min_val, max=max_val)


def predict_labels(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def validate_target(
    target: Optional[torch.Tensor], batch_size: int, num_classes: int, device: torch.device
) -> Optional[torch.Tensor]:
    if target is None:
        return None

    target = target.to(device=device, dtype=torch.long)
    if target.ndim == 0:
        target = target.view(1)

    if target.shape[0] != batch_size:
        raise ValueError(f"Target shape {target.shape[0]} must match batch size {batch_size}")

    if torch.any(target < 0) or torch.any(target >= num_classes):
        raise ValueError(f"Target values must be in range [0, {num_classes})")

    return target


def attack_success(
    logits: torch.Tensor,
    adv_logits: torch.Tensor,
    targeted: bool,
    target: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    adv_pred = predict_labels(adv_logits)
    if targeted is True:
        if target is None:
            raise ValueError("Target labels required for targeted attacks")

        return adv_pred.eq(target)

    base_labels = labels if labels is not None else predict_labels(logits)
    return adv_pred.ne(base_labels)
