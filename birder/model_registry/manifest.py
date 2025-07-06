from typing import Literal
from typing import NotRequired
from typing import TypedDict

FileFormatType = Literal["pt", "safetensors", "pt2", "pts", "ptl"]

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

ModelMetadataType = TypedDict(
    "ModelMetadataType",
    {
        "url": NotRequired[str],
        "description": str,
        "resolution": tuple[int, int],
        "formats": dict[FileFormatType, FormatInfoType],
        "net": NetworkInfoType,
        "backbone": NotRequired[NetworkInfoType],
        "encoder": NotRequired[NetworkInfoType],
        "task": NotRequired[str],
    },
)

REGISTRY_MANIFEST: dict[str, ModelMetadataType] = {}
