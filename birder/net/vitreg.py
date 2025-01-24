"""
Paper "Vision Transformers Need Registers", https://arxiv.org/abs/2309.16588
"""

from typing import Any
from typing import Optional

from birder.model_registry import registry
from birder.net.vit import ViT


class ViTReg4(ViT):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size, num_reg_tokens=4)


registry.register_alias(
    "vitreg4_b32",
    ViTReg4,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "vitreg4_b16",
    ViTReg4,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "vitreg4_l32",
    ViTReg4,
    config={
        "patch_size": 32,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "vitreg4_l16",
    ViTReg4,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "vitreg4_h16",
    ViTReg4,
    config={
        "patch_size": 16,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "vitreg4_h14",
    ViTReg4,
    config={
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "vitreg4_b16_mim_200",
    {
        "url": "https://huggingface.co/birder-project/vitreg4_b16_mim/resolve/main/vitreg4_b16_mim_200.pt",
        "description": (
            "ViTReg4 b16 image encoder pre-trained using Masked Image Modeling (MIM). "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 327.4,
                "sha256": "6b044cd7834293e344309f809070db3fe9ede489478e7549ad96255f9d76b329",
            },
        },
        "net": {"network": "vitreg4_b16", "tag": "mim"},
    },
)
registry.register_weights(
    "vitreg4_b16_mim-intermediate-il-common",
    {
        "url": (
            "https://huggingface.co/birder-project/vitreg4_b16_mim-intermediate-il-common/resolve/"
            "main/vitreg4_b16_mim-intermediate-il-common.pt"
        ),
        "description": (
            "ViTReg4 b16 model with MIM pretraining and intermediate training, "
            "then fine-tuned on the il-common dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 328.7,
                "sha256": "3d1564be46b23081c76aa87c7e90324214b6ced899d4b38d59d1a4154b13f01c",
            },
        },
        "net": {"network": "vitreg4_b16", "tag": "mim-intermediate-il-common"},
    },
)
