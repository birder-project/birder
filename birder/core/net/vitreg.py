"""
Paper "Vision Transformers Need Registers", https://arxiv.org/abs/2309.16588
"""

from typing import Optional

from birder.core.net.vit import ViT
from birder.model_registry import registry


class ViTReg4(ViT):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, num_reg_tokens=4)


registry.register_alias("vit_reg4_b32", ViTReg4, 0)
registry.register_alias("vit_reg4_b16", ViTReg4, 1)
registry.register_alias("vit_reg4_l32", ViTReg4, 2)
registry.register_alias("vit_reg4_l16", ViTReg4, 3)
registry.register_alias("vit_reg4_h14", ViTReg4, 4)
