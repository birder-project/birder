from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.model_registry import Task


class SSLBaseNet(nn.Module):
    default_size: tuple[int, int]
    square_only = False
    task = str(Task.SELF_SUPERVISED_LEARNING)

    def __init__(
        self,
        input_channels: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        if hasattr(self, "net_param") is False:  # Avoid overriding aliases
            self.net_param = net_param
        if hasattr(self, "config") is False:  # Avoid overriding aliases
            self.config = config
        elif config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        if size is not None:
            self.size = size
        else:
            self.size = self.default_size

        assert isinstance(self.size, tuple)
        assert isinstance(self.size[0], int)
        assert isinstance(self.size[1], int)
        if self.square_only is True:
            assert self.size[0] == self.size[1]

    def forward(self, x: torch.Tensor) -> Any:
        raise NotImplementedError
