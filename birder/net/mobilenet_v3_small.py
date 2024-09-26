from typing import Any
from typing import Optional

from birder.net.mobilenet_v3_large import MobileNet_v3_Large


# pylint: disable=invalid-name
class MobileNet_v3_Small(MobileNet_v3_Large):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size, large=False)
