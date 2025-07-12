"""
Paper "Squeeze-and-Excitation Networks", https://arxiv.org/abs/1709.01507
"""

from typing import Any
from typing import Optional

from birder.model_registry import registry
from birder.net.resnet_v2 import ResNet_v2


# pylint: disable=invalid-name
class SE_ResNet_v2(ResNet_v2):
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


registry.register_model_config(
    "se_resnet_v2_18",
    SE_ResNet_v2,
    config={"bottle_neck": False, "filter_list": [64, 64, 128, 256, 512], "units": [2, 2, 2, 2]},
)
registry.register_model_config(
    "se_resnet_v2_34",
    SE_ResNet_v2,
    config={"bottle_neck": False, "filter_list": [64, 64, 128, 256, 512], "units": [3, 4, 6, 3]},
)
registry.register_model_config(
    "se_resnet_v2_50",
    SE_ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 4, 6, 3]},
)
registry.register_model_config(
    "se_resnet_v2_101",
    SE_ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 4, 23, 3]},
)
registry.register_model_config(
    "se_resnet_v2_152",
    SE_ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 8, 36, 3]},
)
registry.register_model_config(
    "se_resnet_v2_200",
    SE_ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 24, 36, 3]},
)
registry.register_model_config(
    "se_resnet_v2_269",
    SE_ResNet_v2,
    config={"bottle_neck": True, "filter_list": [64, 256, 512, 1024, 2048], "units": [3, 30, 48, 8]},
)
