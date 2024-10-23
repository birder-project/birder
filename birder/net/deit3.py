"""
Paper "DeiT III: Revenge of the ViT", https://arxiv.org/abs/2204.07118
"""

from typing import Any
from typing import Optional

from birder.model_registry import registry
from birder.net.deit import DeiT


class DeiT3(DeiT):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_channels, num_classes, net_param=net_param, size=size, config=config, pos_embed_class=False
        )


registry.register_alias(
    "deit3_t16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "deit3_s16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "deit3_b16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "deit3_t16_il-common",
    {
        "description": "DeiT3 tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.7,
                "sha256": "a883c5240cd3f8b6b003218e6016e18d01f8a5243481acbeeb3cbc3bf68a76af",
            }
        },
        "net": {"network": "deit3_t16", "tag": "il-common"},
    },
)
