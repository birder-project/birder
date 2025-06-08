from typing import Any
from typing import Optional
from typing import TypedDict

import torch
from torch import nn

from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.base import DataShapeType

SSLSignatureType = TypedDict(
    "SSLSignatureType",
    {
        "inputs": list[DataShapeType],
        "outputs": list[DataShapeType],
    },
)


def get_ssl_signature(input_shape: tuple[int, ...]) -> SSLSignatureType:
    return {
        "inputs": [{"data_shape": [0, *input_shape[1:]]}],
        "outputs": [{"data_shape": [0]}],
    }


class SSLBaseNet(nn.Module):
    auto_register = True
    square_only = False
    task = str(Task.SELF_SUPERVISED_LEARNING)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.auto_register is False:
            # Exclude networks with custom config (initialized only with aliases)
            return

        registry.register_model(cls.__name__.lower(), cls)

    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.input_channels = backbone.input_channels
        self.backbone = backbone
        self.net_param = None  # For compatibility with other base nets

        if hasattr(self, "config") is False:  # Avoid overriding aliases
            self.config = config
        elif config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        if size is not None:
            self.size = size
        else:
            self.size = self.backbone.size

        assert isinstance(self.size, tuple)
        assert isinstance(self.size[0], int)
        assert isinstance(self.size[1], int)
        if self.square_only is True:
            assert self.size[0] == self.size[1]

        self.backbone.adjust_size(self.size)

    def forward(self, x: torch.Tensor) -> Any:
        raise NotImplementedError
