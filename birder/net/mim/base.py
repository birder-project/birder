from typing import Any
from typing import Optional
from typing import TypedDict

import torch
from torch import nn

from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DataShapeType
from birder.net.base import PreTrainEncoder

MIMSignatureType = TypedDict(
    "MIMSignatureType",
    {
        "inputs": list[DataShapeType],
        "outputs": list[DataShapeType],
    },
)


def get_mim_signature(input_shape: tuple[int, ...]) -> MIMSignatureType:
    return {
        "inputs": [{"data_shape": [0, *input_shape[1:]]}],
        "outputs": [{"data_shape": [0, *input_shape[1:]]}],
    }


class MIMBaseNet(nn.Module):
    default_size: tuple[int, int]
    task = str(Task.MASKED_IMAGE_MODELING)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        registry.register_model(cls.__name__.lower(), cls)

    def __init__(
        self,
        encoder: PreTrainEncoder,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.input_channels = encoder.input_channels
        self.encoder = encoder
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError
