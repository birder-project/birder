from typing import Optional

from birder.core.net.swin_transformer_v2 import Swin_Transformer_v2


# pylint: disable=invalid-name
class Swin_Transformer_v2_w2(Swin_Transformer_v2):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, window_scale_factor=2)
