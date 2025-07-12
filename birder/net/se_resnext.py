"""
Paper "Squeeze-and-Excitation Networks", https://arxiv.org/abs/1709.01507
"""

from typing import Any
from typing import Optional

from birder.model_registry import registry
from birder.net.resnext import ResNeXt


# pylint: disable=invalid-name
class SE_ResNeXt(ResNeXt):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            input_channels, num_classes, net_param=net_param, config=config, size=size, squeeze_excitation=True
        )


registry.register_model_config("se_resnext_50", SE_ResNeXt, config={"units": [3, 4, 6, 3]})
registry.register_model_config("se_resnext_101", SE_ResNeXt, config={"units": [3, 4, 23, 3]})
registry.register_model_config("se_resnext_152", SE_ResNeXt, config={"units": [3, 8, 36, 3]})
