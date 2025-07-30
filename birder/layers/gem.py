"""
Generalized Mean (GeM) Pooling Layers

These modules implement Generalized Mean Pooling as described in the paper
"Fine-tuning CNN Image Retrieval with No Human Annotation"
https://arxiv.org/abs/1711.02512

Two variants are provided:
1. FixedGeMPool2d: Performs global GeM pooling with a fixed, non-trainable pooling parameter p.
2. GeMPool2d: Performs global GeM pooling with a trainable pooling parameter "p".
   This is often used as the final pooling layer in image retrieval networks.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FixedGeMPool2d(nn.Module):
    def __init__(self, pooling_param: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = pooling_param
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2)
        mean = x.clamp(min=self.eps).pow(self.p).mean(dim=2)

        return mean.pow(1.0 / self.p)


class GeMPool2d(nn.Module):
    def __init__(self, pooling_param: float, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * pooling_param)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1)))  # pylint:disable=not-callable
        return mean.pow(1.0 / self.p).squeeze(dim=(-2, -1))
