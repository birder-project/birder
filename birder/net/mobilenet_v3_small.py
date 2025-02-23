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
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size, large=False)


registry.register_weights(
    "mobilenet_v3_small_1_il-common",
    {
        "description": "MobileNet v3 small (1.0 multiplier) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 7.4,
                "sha256": "ac53227f7513fd0c0b5204ee57403de2ab6c74c4e4d1061b9168596c6b5cea48",
            }
        },
        "net": {"network": "mobilenet_v3_small", "net_param": 1, "tag": "il-common"},
    },
)
