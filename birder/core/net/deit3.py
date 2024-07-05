"""
Paper "DeiT III: Revenge of the ViT", https://arxiv.org/abs/2204.07118
"""

from typing import Optional

from birder.core.net.deit import DeiT


class DeiT3(DeiT):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, pos_embed_class=False)