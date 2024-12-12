from typing import Literal
from typing import TypedDict

from typing_extensions import NotRequired

FormatInfoType = TypedDict(
    "FormatInfoType",
    {"file_size": float, "sha256": str},
)

NetworkInfoType = TypedDict(
    "NetworkInfoType",
    {
        "network": str,
        "net_param": NotRequired[float],
        "tag": NotRequired[str],
        "reparameterized": NotRequired[bool],
    },
)

ModelInfoType = TypedDict(
    "ModelInfoType",
    {
        "url": NotRequired[str],
        "description": str,
        "resolution": tuple[int, int],
        "formats": dict[Literal["pt", "pt2", "pts", "ptl", "safetensors"], FormatInfoType],
        "net": NetworkInfoType,
        "backbone": NotRequired[NetworkInfoType],
    },
)

REGISTRY_MANIFEST: dict[str, ModelInfoType] = {}
