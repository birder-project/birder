"""
Paper "Vision Transformers Need Registers", https://arxiv.org/abs/2309.16588
"""

from typing import Optional

from birder.core.net.base import create_alias
from birder.core.net.vit import ViT


class ViTReg4(ViT):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, num_reg_tokens=4)


create_alias("vit_reg4_b_32", ViTReg4, 0)
create_alias("vit_reg4_b_16", ViTReg4, 1)
create_alias("vit_reg4_l_32", ViTReg4, 2)
create_alias("vit_reg4_l_16", ViTReg4, 3)
create_alias("vit_reg4_h_14", ViTReg4, 4)
