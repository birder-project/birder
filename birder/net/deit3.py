"""
Paper "DeiT III: Revenge of the ViT", https://arxiv.org/abs/2204.07118
"""

from typing import Optional

from birder.model_registry import registry
from birder.net.deit import DeiT


class DeiT3(DeiT):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size, pos_embed_class=False)


registry.register_alias("deit3_t16", DeiT3, 0)
registry.register_alias("deit3_s16", DeiT3, 1)
registry.register_alias("deit3_b16", DeiT3, 2)

registry.register_weights(
    "deit3_t16_il-common",
    {
        "description": "DeiT3 tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.7,
                "sha256": "26636f54c67962d2a0f8f8fca92c7619112776bf404910a89528b03323b4506a",
            }
        },
        "net": {"network": "deit_t16", "tag": "il-common"},
    },
)
