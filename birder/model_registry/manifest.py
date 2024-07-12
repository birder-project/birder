from typing import TypedDict

ModelInfoType = TypedDict(
    "ModelInfoType", {"description": str, "file_size": float, "formats": list[str], "sha256": str}
)

REGISTRY_MANIFEST: dict[str, ModelInfoType] = {
    "mobilenet_v3_1": {
        "description": "Just a placeholder network",
        "file_size": 17.3,
        "formats": ["pt"],
        "sha256": "d014717fbfef85828acdb85076b018af72bab9ac48ff367e10426259b4360d9d",
    },
    "mobilenet_v3_1_quantized": {
        "description": "Just a placeholder network (quantized for CPU)",
        "file_size": 4.5,
        "formats": ["pts"],
        "sha256": "3d4c9621077267f0e3e8ad7a5fb05030a3fb9eab3272c89c2caf08e5cd661697",
    },
}
