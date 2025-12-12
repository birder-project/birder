import random
from typing import Any

import torch
from PIL import Image
from torchvision import tv_tensors


# pylint: disable=too-many-locals
def mosaic_random_center(
    images: list[Image.Image],
    targets: list[dict[str, Any]],
    output_size: tuple[int, int],
    fill_value: int | tuple[int, int, int],
    min_box_area: float = 0.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Create a mosaic augmentation by combining 4 images into a single image.

    This augmentation places 4 images on a canvas, meeting at a randomly selected
    center point. Each image is scaled to fit, cropped as needed, and their bounding
    boxes are transformed accordingly.

    Parameters
    ----------
    images
        List of exactly 4 PIL images. Order: top-left, top-right, bottom-left, bottom-right.
    targets
        List of exactly 4 target dicts, each containing:
        - "boxes": Tensor of shape (N, 4) in XYXY format.
        - "labels": Tensor of shape (N,) with class indices.
    output_size
        Target output size as (width, height).
    fill_value
        Background fill color. Can be a single int (grayscale) or RGB tuple.
    min_box_area
        Minimum box area in pixels to keep after clipping. Boxes smaller than this are removed.

    Returns
    -------
    tuple[Image.Image, dict[str, Any]]
        The mosaic image (RGB) and merged target dict with "boxes" and "labels".
    """

    assert len(images) == 4 and len(targets) == 4

    (output_w, output_h) = output_size

    fill_color: tuple[int, int, int]
    if isinstance(fill_value, int):
        fill_color = (fill_value, fill_value, fill_value)
    else:
        fill_color = fill_value

    mosaic_img = Image.new("RGB", (output_w, output_h), color=fill_color)

    xc = int(random.uniform(output_w * 0.25, output_w * 0.75))
    yc = int(random.uniform(output_h * 0.25, output_h * 0.75))

    quadrants = [
        (0, 0, xc, yc),
        (xc, 0, output_w, yc),
        (0, yc, xc, output_h),
        (xc, yc, output_w, output_h),
    ]

    all_boxes: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for i, (img, target) in enumerate(zip(images, targets)):
        if img.mode != "RGB":
            img = img.convert("RGB")

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if isinstance(boxes, tv_tensors.BoundingBoxes):
                boxes = boxes.data.clone().float()
            else:
                boxes = boxes.clone().float()
            labels = target["labels"].clone()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        (w, h) = img.size
        (q_x1, q_y1, q_x2, q_y2) = quadrants[i]

        scale = min(output_w / w, output_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Canvas coordinates (where to paste)
        x1a = max(q_x1, q_x2 - new_w) if i in (0, 2) else q_x1
        y1a = max(q_y1, q_y2 - new_h) if i in (0, 1) else q_y1
        x2a = min(q_x2, x1a + new_w)
        y2a = min(q_y2, y1a + new_h)

        # Source image coordinates (what to crop)
        if i == 0:
            x1b = new_w - (x2a - x1a)
            y1b = new_h - (y2a - y1a)
        elif i == 1:
            x1b = 0
            y1b = new_h - (y2a - y1a)
        elif i == 2:
            x1b = new_w - (x2a - x1a)
            y1b = 0
        else:
            x1b = 0
            y1b = 0

        x2b = x1b + (x2a - x1a)
        y2b = y1b + (y2a - y1a)
        mosaic_img.paste(img.crop((x1b, y1b, x2b, y2b)), (x1a, y1a))

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= new_w / w
            boxes[:, [1, 3]] *= new_h / h
            offset_x = x1a - x1b
            offset_y = y1a - y1b
            boxes[:, 0] += offset_x
            boxes[:, 2] += offset_x
            boxes[:, 1] += offset_y
            boxes[:, 3] += offset_y
            all_boxes.append(boxes)
            all_labels.append(labels)

    if len(all_boxes) > 0:
        boxes = torch.concat(all_boxes, dim=0)
        labels = torch.concat(all_labels, dim=0)

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, output_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, output_h)

        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        valid = (ws > 0) & (hs > 0) & ((ws * hs) > min_box_area)
        boxes = boxes[valid]
        labels = labels[valid]

    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

    boxes = tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(output_h, output_w))

    return (mosaic_img, {"boxes": boxes, "labels": labels})


# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def mosaic_fixed_grid(
    images: list[Image.Image],
    targets: list[dict[str, Any]],
    output_size: tuple[int, int],
    fill_value: int | tuple[int, int, int],
    min_visibility: float = 0.05,
    crop_to_square: bool = False,
    max_aspect_ratio: float | None = 2.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Create a grid-based mosaic augmentation by combining 4 images into a 2x2 grid.

    This augmentation arranges 4 images in a fixed 2x2 grid layout. Images are first
    resized/padded to squares, then assembled into a larger grid which is randomly
    cropped to the output size. Based on the approach from YOLOv4.

    Reference: https://github.com/prymeka/mosaic-augmentation-pytorch/blob/main/mosaic.py (MIT License)

    Parameters
    ----------
    images
        List of exactly 4 PIL images. Order: top-left, top-right, bottom-left, bottom-right.
    targets
        List of exactly 4 target dicts, each containing:
        - "boxes": Tensor of shape (N, 4) in XYXY format.
        - "labels": Tensor of shape (N,) with class indices.
    output_size
        Target output size as (width, height).
    fill_value
        Background fill color. Can be a single int (grayscale) or RGB tuple.
    min_visibility
        Minimum visibility ratio (area_after / area_before) to keep a box after cropping.
    crop_to_square
        If True, crop all images to square before assembly. Otherwise, pad to square.
    max_aspect_ratio
        Maximum aspect ratio before cropping to reduce it. Ignored if crop_to_square is True.

    Returns
    -------
    tuple[Image.Image, dict[str, Any]]
        The mosaic image (RGB) and merged target dict with "boxes" and "labels".
    """

    assert len(images) == 4 and len(targets) == 4

    (output_w, output_h) = output_size

    fill_color: tuple[int, int, int]
    if isinstance(fill_value, int):
        fill_color = (fill_value, fill_value, fill_value)
    else:
        fill_color = fill_value

    # Grid size (larger than output to allow cropping)
    min_possible_image_area = 0.25
    grid_size = int(max(output_w, output_h) * (2 - min_possible_image_area**0.5))
    image_size = grid_size // 2

    grid_img = Image.new("RGB", (grid_size, grid_size), color=fill_color)
    positions = [
        (0, 0),
        (image_size, 0),
        (0, image_size),
        (image_size, image_size),
    ]

    all_boxes: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for i, (img, target) in enumerate(zip(images, targets)):
        if img.mode != "RGB":
            img = img.convert("RGB")

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if isinstance(boxes, tv_tensors.BoundingBoxes):
                boxes = boxes.data.clone().float()
            else:
                boxes = boxes.clone().float()
            labels = target["labels"].clone()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        (w, h) = img.size

        if crop_to_square:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

            if len(boxes) > 0:
                boxes[:, 0] -= left
                boxes[:, 2] -= left
                boxes[:, 1] -= top
                boxes[:, 3] -= top
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, min_dim)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, min_dim)

            (w, h) = img.size

        elif max_aspect_ratio is not None:
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > max_aspect_ratio:
                if w > h:
                    new_w = int(h * max_aspect_ratio)
                    if new_w < w:
                        left = (w - new_w) // 2
                        img = img.crop((left, 0, left + new_w, h))
                        if len(boxes) > 0:
                            boxes[:, 0] -= left
                            boxes[:, 2] -= left
                            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)

                        (w, h) = img.size
                else:
                    new_h = int(w * max_aspect_ratio)
                    if new_h < h:
                        top = (h - new_h) // 2
                        img = img.crop((0, top, w, top + new_h))
                        if len(boxes) > 0:
                            boxes[:, 1] -= top
                            boxes[:, 3] -= top
                            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

                        (w, h) = img.size

        # Pad to square (content towards center, padding towards edges)
        max_dim = max(w, h)
        pad_w = max_dim - w
        pad_h = max_dim - h

        if i == 0:
            paste_x, paste_y = pad_w, pad_h
        elif i == 1:
            paste_x, paste_y = 0, pad_h
        elif i == 2:
            paste_x, paste_y = pad_w, 0
        else:
            paste_x, paste_y = 0, 0

        if w != h:
            padded_img = Image.new("RGB", (max_dim, max_dim), color=fill_color)
            padded_img.paste(img, (paste_x, paste_y))
            img = padded_img

            if len(boxes) > 0:
                boxes[:, 0] += paste_x
                boxes[:, 2] += paste_x
                boxes[:, 1] += paste_y
                boxes[:, 3] += paste_y

        scale = image_size / max_dim
        img = img.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)

        if len(boxes) > 0:
            boxes *= scale

        (px, py) = positions[i]
        grid_img.paste(img, (px, py))

        if len(boxes) > 0:
            boxes[:, 0] += px
            boxes[:, 2] += px
            boxes[:, 1] += py
            boxes[:, 3] += py
            all_boxes.append(boxes)
            all_labels.append(labels)

    max_x = grid_size - output_w
    max_y = grid_size - output_h
    crop_x = random.randint(0, max(0, max_x))
    crop_y = random.randint(0, max(0, max_y))

    mosaic_img = grid_img.crop((crop_x, crop_y, crop_x + output_w, crop_y + output_h))

    if len(all_boxes) > 0:
        boxes = torch.concat(all_boxes, dim=0)
        labels = torch.concat(all_labels, dim=0)

        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        orig_areas = ws * hs

        boxes[:, 0] -= crop_x
        boxes[:, 2] -= crop_x
        boxes[:, 1] -= crop_y
        boxes[:, 3] -= crop_y
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, output_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, output_h)

        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        valid = (ws > 0) & (hs > 0) & ((ws * hs) / (orig_areas + 1e-6) >= min_visibility)
        boxes = boxes[valid]
        labels = labels[valid]

    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

    boxes = tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(output_h, output_w))

    return (mosaic_img, {"boxes": boxes, "labels": labels})
