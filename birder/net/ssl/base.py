from typing import Any

import torch
from torch import nn

from birder.model_registry import Task


class SSLBaseNet(nn.Module):
    default_size: tuple[int, int]
    task = str(Task.SELF_SUPERVISED_LEARNING)

    def __init__(self) -> None:
        super().__init__()

        # Just ensure some attributes are defined for compatibility
        self.config = None
        self.net_param = None

    def forward(self, x: torch.Tensor) -> Any:
        raise NotImplementedError
