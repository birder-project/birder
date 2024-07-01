from typing import Optional

import torch
from torch import nn

from birder.core.net.base import PreTrainEncoder


class PreTrainBaseNet(nn.Module):
    default_size: int
    task = "image_pretraining"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        _REGISTERED_PRETRAIN_NETWORKS[cls.__name__.lower()] = cls

    def __init__(
        self,
        encoder: PreTrainEncoder,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_channels = encoder.input_channels
        self.encoder = encoder
        if hasattr(self, "net_param") is False:  # Avoid overriding aliases
            self.net_param = net_param

        if size is not None:
            self.size = size

        else:
            self.size = self.default_size

        assert isinstance(self.size, int)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def pretrain_net_factory(
    name: str,
    encoder: PreTrainBaseNet,
    net_param: Optional[float] = None,
    size: Optional[int] = None,
) -> PreTrainBaseNet:
    return _REGISTERED_PRETRAIN_NETWORKS[name](encoder, net_param, size)


_REGISTERED_PRETRAIN_NETWORKS: dict[str, type[PreTrainBaseNet]] = {}
