from typing import Any
from typing import Optional

from birder.model_registry import registry
from birder.net.mobilenet_v3_large import MobileNet_v3_Large


# pylint: disable=invalid-name
class MobileNet_v3_Small(MobileNet_v3_Large):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size, large=False)


registry.register_model_config("mobilenet_v3_small_0_25", MobileNet_v3_Small, config={"alpha": 0.25})
registry.register_model_config("mobilenet_v3_small_0_5", MobileNet_v3_Small, config={"alpha": 0.5})
registry.register_model_config("mobilenet_v3_small_0_75", MobileNet_v3_Small, config={"alpha": 0.75})
registry.register_model_config("mobilenet_v3_small_1_0", MobileNet_v3_Small, config={"alpha": 1.0})
registry.register_model_config("mobilenet_v3_small_1_25", MobileNet_v3_Small, config={"alpha": 1.25})
registry.register_model_config("mobilenet_v3_small_1_5", MobileNet_v3_Small, config={"alpha": 1.5})
registry.register_model_config("mobilenet_v3_small_1_75", MobileNet_v3_Small, config={"alpha": 1.75})
registry.register_model_config("mobilenet_v3_small_2_0", MobileNet_v3_Small, config={"alpha": 2.0})

registry.register_weights(
    "mobilenet_v3_small_1_0_il-common",
    {
        "description": "MobileNet v3 small (1.0 multiplier) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 7.4,
                "sha256": "ac53227f7513fd0c0b5204ee57403de2ab6c74c4e4d1061b9168596c6b5cea48",
            }
        },
        "net": {"network": "mobilenet_v3_small_1_0", "tag": "il-common"},
    },
)
