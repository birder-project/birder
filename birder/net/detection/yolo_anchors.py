"""
Shared YOLO anchor definitions and helpers.
"""

import json
from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import TypedDict
from typing import overload

AnchorPair = tuple[float, float]
AnchorGroup = list[AnchorPair]
AnchorGroups = list[AnchorGroup]
AnchorLike = AnchorGroups | AnchorGroup

# Default anchors from yolo.cfg (COCO dataset), in grid units
YOLO_V2_ANCHORS: AnchorGroup = [
    (0.57273, 0.677385),
    (1.87446, 2.06253),
    (3.33843, 5.47434),
    (7.88282, 3.52778),
    (9.77052, 9.16828),
]

# Default anchors from YOLO v3 paper (sorted by area, small to large)
# These values are in absolute pixels (width, height) computed using K-Means
# on the COCO dataset with a reference input size of 416x416.
YOLO_V3_ANCHORS: AnchorGroups = [
    [(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)],  # Small objects (stride 8)
    [(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)],  # Medium objects (stride 16)
    [(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)],  # Large objects (stride 32)
]

# Default anchors from YOLO v4 (COCO), in pixels
YOLO_V4_ANCHORS: AnchorGroups = [
    [(12.0, 16.0), (19.0, 36.0), (40.0, 28.0)],  # Small
    [(36.0, 75.0), (76.0, 55.0), (72.0, 146.0)],  # Medium
    [(142.0, 110.0), (192.0, 243.0), (459.0, 401.0)],  # Large
]

# Default anchors from YOLO v4 Tiny (COCO), in pixels
YOLO_V4_TINY_ANCHORS: AnchorGroups = [
    [(10.0, 14.0), (23.0, 27.0), (37.0, 58.0)],  # Medium
    [(81.0, 82.0), (135.0, 169.0), (344.0, 319.0)],  # Large
]


class AnchorPreset(TypedDict):
    anchors: AnchorLike
    format: Literal["grid", "pixels"]
    size: tuple[int, int]
    strides: NotRequired[Sequence[int]]


ANCHOR_PRESETS: dict[str, AnchorPreset] = {
    "yolo_v2": {"anchors": YOLO_V2_ANCHORS, "format": "grid", "size": (416, 416), "strides": (32,)},
    "yolo_v3": {"anchors": YOLO_V3_ANCHORS, "format": "pixels", "size": (416, 416)},
    "yolo_v4": {"anchors": YOLO_V4_ANCHORS, "format": "pixels", "size": (608, 608)},
    "yolo_v4_tiny": {"anchors": YOLO_V4_TINY_ANCHORS, "format": "pixels", "size": (416, 416)},
}


@overload
def scale_anchors(anchors: AnchorGroup, from_size: tuple[int, int], to_size: tuple[int, int]) -> AnchorGroup: ...


@overload
def scale_anchors(anchors: AnchorGroups, from_size: tuple[int, int], to_size: tuple[int, int]) -> AnchorGroups: ...


def scale_anchors(anchors: AnchorLike, from_size: tuple[int, int], to_size: tuple[int, int]) -> AnchorLike:
    (anchor_groups, single) = _normalize_anchor_groups(anchors)

    if from_size == to_size:
        # Avoid aliasing default anchors in case they are mutated later
        scaled: AnchorGroups = [list(group) for group in anchor_groups]
        if single is True:
            return scaled[0]

        return scaled

    scale_h = to_size[0] / from_size[0]
    scale_w = to_size[1] / from_size[1]
    scaled = [[(w * scale_w, h * scale_h) for (w, h) in group] for group in anchor_groups]

    if single is True:
        return scaled[0]

    return scaled


@overload
def pixels_to_grid(anchors: AnchorGroup, strides: Sequence[int]) -> AnchorGroup: ...


@overload
def pixels_to_grid(anchors: AnchorGroups, strides: Sequence[int]) -> AnchorGroups: ...


def pixels_to_grid(anchors: AnchorLike, strides: Sequence[int]) -> AnchorLike:
    (anchor_groups, single) = _normalize_anchor_groups(anchors)
    if len(anchor_groups) != len(strides):
        raise ValueError("strides must provide one value per anchor scale")

    converted: AnchorGroups = []
    for group, stride in zip(anchor_groups, strides):
        converted.append([(w / stride, h / stride) for (w, h) in group])

    if single is True:
        return converted[0]

    return converted


@overload
def grid_to_pixels(anchors: AnchorGroup, strides: Sequence[int]) -> AnchorGroup: ...


@overload
def grid_to_pixels(anchors: AnchorGroups, strides: Sequence[int]) -> AnchorGroups: ...


def grid_to_pixels(anchors: AnchorLike, strides: Sequence[int]) -> AnchorLike:
    (anchor_groups, single) = _normalize_anchor_groups(anchors)
    if len(anchor_groups) != len(strides):
        raise ValueError("strides must provide one value per anchor scale")

    converted: AnchorGroups = []
    for group, stride in zip(anchor_groups, strides):
        converted.append([(w * stride, h * stride) for (w, h) in group])

    if single is True:
        return converted[0]

    return converted


def _normalize_anchor_groups(anchors: AnchorLike) -> tuple[AnchorGroups, bool]:
    if len(anchors) > 0 and _is_anchor_pair(anchors[0]) is True:
        return ([anchors], True)  # type: ignore[list-item]

    return (anchors, False)  # type: ignore[return-value]


def _is_anchor_pair(value: Any) -> bool:
    if not isinstance(value, Sequence) or len(value) != 2:
        return False

    return all(isinstance(item, (float, int)) for item in value)


def _resolve_anchors(
    preset: str, *, anchor_format: str, model_size: tuple[int, int], model_strides: Sequence[int]
) -> AnchorLike:
    if preset.endswith(".json") is True:
        with open(preset, "r", encoding="utf-8") as handle:
            preset_spec = json.load(handle)
    else:
        if preset not in ANCHOR_PRESETS:
            raise ValueError(f"Unknown anchor preset: {preset}")

        preset_spec = ANCHOR_PRESETS[preset]

    anchors = preset_spec["anchors"]
    preset_size = tuple(preset_spec["size"])
    preset_format = preset_spec["format"]
    if preset_format == "grid":
        if "strides" not in preset_spec:
            raise ValueError("Preset is missing strides required for grid anchors")

        preset_strides = preset_spec["strides"]
        anchors = grid_to_pixels(anchors, preset_strides)

    anchors = scale_anchors(anchors, preset_size, model_size)
    if anchor_format == "pixels":
        return anchors

    if anchor_format == "grid":
        return pixels_to_grid(anchors, model_strides)

    raise ValueError(f"Unsupported anchor format: {anchor_format}")


def resolve_anchor_group(
    preset: str, *, anchor_format: str, model_size: tuple[int, int], model_strides: Sequence[int]
) -> AnchorGroup:
    anchors = _resolve_anchors(preset, anchor_format=anchor_format, model_size=model_size, model_strides=model_strides)
    (anchor_groups, single) = _normalize_anchor_groups(anchors)
    if single is False:
        raise ValueError("Expected a single anchor group for this model")

    return anchor_groups[0]


def resolve_anchor_groups(
    preset: str, *, anchor_format: str, model_size: tuple[int, int], model_strides: Sequence[int]
) -> AnchorGroups:
    anchors = _resolve_anchors(preset, anchor_format=anchor_format, model_size=model_size, model_strides=model_strides)
    (anchor_groups, single) = _normalize_anchor_groups(anchors)
    if single is True:
        raise ValueError("Expected multiple anchor groups for this model")

    return anchor_groups
