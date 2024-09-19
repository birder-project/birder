"""
Paper "Squeeze-and-Excitation Networks", https://arxiv.org/abs/1709.01507
"""

from typing import Optional

from birder.net.resnet_v2 import ResNet_v2


# pylint: disable=invalid-name
class SE_ResNet_v2(ResNet_v2):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, squeeze_excitation=True)
