"""
Paper "Squeeze-and-Excitation Networks", https://arxiv.org/abs/1709.01507
"""

from typing import Optional

from birder.net.resnext import ResNeXt


# pylint: disable=invalid-name
class SE_ResNeXt(ResNeXt):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, squeeze_excitation=True)
