from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.amp
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from birder.conf import settings
from birder.data.collators.detection import batch_images
from birder.data.transforms.detection import InferenceTransform
from birder.inference.wbf import fuse_detections_wbf
from birder.net.base import make_divisible


def _normalize_image_sizes(inputs: torch.Tensor, image_sizes: Optional[list[list[int]]]) -> list[list[int]]:
    if image_sizes is not None:
        return image_sizes

    (_, _, height, width) = inputs.shape
    return [[height, width] for _ in range(inputs.size(0))]


def _hflip_inputs(inputs: torch.Tensor, image_sizes: list[list[int]]) -> torch.Tensor:
    # Detection collator pads on the right/bottom, so flip only the valid region to keep padding aligned.
    flipped = inputs.clone()
    for idx, (height, width) in enumerate(image_sizes):
        flipped[idx, :, :height, :width] = torch.flip(inputs[idx, :, :height, :width], dims=[2])

    return flipped


def _resize_batch(
    inputs: torch.Tensor, image_sizes: list[list[int]], scale: float, size_divisible: int
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    resized_images: list[torch.Tensor] = []
    for idx, (height, width) in enumerate(image_sizes):
        target_h = make_divisible(height * scale, size_divisible)
        target_w = make_divisible(width * scale, size_divisible)
        image = inputs[idx, :, :height, :width]
        resized = F.interpolate(image.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False)
        resized_images.append(resized.squeeze(0))

    return batch_images(resized_images, size_divisible)


def _rescale_boxes(boxes: torch.Tensor, from_size: list[int], to_size: list[int]) -> torch.Tensor:
    scale_w = to_size[1] / from_size[1]
    scale_h = to_size[0] / from_size[0]
    scale = boxes.new_tensor([scale_w, scale_h, scale_w, scale_h])
    return boxes * scale


def _rescale_detections(
    detections: list[dict[str, torch.Tensor]],
    from_sizes: list[list[int]],
    to_sizes: list[list[int]],
) -> list[dict[str, torch.Tensor]]:
    for idx, (detection, from_size, to_size) in enumerate(zip(detections, from_sizes, to_sizes)):
        boxes = detection["boxes"]
        if boxes.numel() == 0:
            continue

        detections[idx]["boxes"] = _rescale_boxes(boxes, from_size, to_size)

    return detections


def _invert_hflip_boxes(boxes: torch.Tensor, image_size: list[int]) -> torch.Tensor:
    width = boxes.new_tensor(image_size[1])
    x1 = boxes[:, 0]
    x2 = boxes[:, 2]
    flipped = boxes.clone()
    flipped[:, 0] = width - x2
    flipped[:, 2] = width - x1

    return flipped


def _invert_detections(
    detections: list[dict[str, torch.Tensor]], image_sizes: list[list[int]]
) -> list[dict[str, torch.Tensor]]:
    for idx, (detection, image_size) in enumerate(zip(detections, image_sizes)):
        boxes = detection["boxes"]
        if boxes.numel() == 0:
            continue

        detections[idx]["boxes"] = _invert_hflip_boxes(boxes, image_size)

    return detections


def infer_image(
    net: torch.nn.Module | torch.ScriptModule,
    sample: Image.Image | str,
    transform: Callable[..., torch.Tensor],
    tta: bool = False,
    device: Optional[torch.device] = None,
    score_threshold: Optional[float] = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Perform inference on a single image

    This convenience function allows for quick, one-off detection of an image.

    Raises
    ------
    TypeError
        If the sample is neither a string nor a PIL Image object.
    """

    image: Image.Image
    if isinstance(sample, str):
        image = Image.open(sample)
    elif isinstance(sample, Image.Image):
        image = sample
    else:
        raise TypeError("Unknown sample type")

    if device is None:
        device = torch.device("cpu")

    input_tensor = transform(image).unsqueeze(dim=0).to(device)
    detections = infer_batch(net, input_tensor, tta=tta, **kwargs)
    if score_threshold is not None:
        for i, detection in enumerate(detections):
            idxs = torch.where(detection["scores"] > score_threshold)
            detections[i]["scores"] = detection["scores"][idxs]
            detections[i]["boxes"] = detection["boxes"][idxs]
            detections[i]["labels"] = detection["labels"][idxs]

    detections = InferenceTransform.postprocess(
        detections, [input_tensor.shape[2:]], [image.size[::-1]]  # type: ignore[list-item]
    )

    return detections[0]


def infer_batch(
    net: torch.nn.Module | torch.ScriptModule,
    inputs: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    image_sizes: Optional[list[list[int]]] = None,
    tta: bool = False,
    **kwargs: Any,
) -> list[dict[str, torch.Tensor]]:
    if tta is False:
        (detections, _) = net(inputs, masks=masks, image_sizes=image_sizes, **kwargs)
        return detections  # type: ignore[no-any-return]

    normalized_sizes = _normalize_image_sizes(inputs, image_sizes)
    detections_list: list[list[dict[str, torch.Tensor]]] = []

    for scale in (0.8, 1.0, 1.2):
        (scaled_inputs, scaled_masks, scaled_sizes) = _resize_batch(inputs, normalized_sizes, scale, size_divisible=32)
        (detections, _) = net(scaled_inputs, masks=scaled_masks, image_sizes=scaled_sizes, **kwargs)
        detections = _rescale_detections(detections, scaled_sizes, normalized_sizes)
        detections_list.append(detections)

        flipped_inputs = _hflip_inputs(scaled_inputs, scaled_sizes)
        (flipped_detections, _) = net(flipped_inputs, masks=scaled_masks, image_sizes=scaled_sizes, **kwargs)
        flipped_detections = _invert_detections(flipped_detections, scaled_sizes)
        flipped_detections = _rescale_detections(flipped_detections, scaled_sizes, normalized_sizes)
        detections_list.append(flipped_detections)

    return fuse_detections_wbf(detections_list, iou_thr=0.55, conf_type="avg")


def infer_dataloader(
    device: torch.device,
    net: torch.nn.Module | torch.ScriptModule,
    dataloader: DataLoader,
    tta: bool = False,
    model_dtype: torch.dtype = torch.float32,
    amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    num_samples: Optional[int] = None,
    batch_callback: Optional[
        Callable[[list[str], torch.Tensor, list[dict[str, torch.Tensor]], list[dict[str, Any]], list[list[int]]], None]
    ] = None,
) -> tuple[list[str], list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """
    Perform inference on a DataLoader using a given neural network.

    This function runs inference on a dataset provided through a DataLoader,
    optionally using mixed precision (amp).
    All returned detections and targets are transformed to original
    image coordinates regardless of the model's inference resolution.

    Parameters
    ----------
    device
        The device to run the inference on.
    net
        The model to use for inference.
    dataloader
        The DataLoader containing the dataset to perform inference on.
    tta
        Run inference with multi-scale and horizontal flip test time augmentation and fuse results with WBF.
    model_dtype
        The base dtype to use.
    amp
        Whether to use automatic mixed precision.
    amp_dtype
        The mixed precision dtype.
    num_samples
        The total number of samples in the dataloader.
    batch_callback
        A function to be called after each batch is processed. If provided, it
        should accept four arguments:
        - list[str]: A list of file paths for the current batch
        - torch.Tensor: The input tensor for the current batch
        - list[dict[str, torch.Tensor]]: The detections for the current batch
        - list[dict[str, Any]]: A list of targets for the current batch
        - list[list[int]]: The image sizes for the current batch

    Returns
    -------
    A tuple containing three elements:
    - list[str]: A list of all processed file paths.
    - list[dict[str, torch.Tensor]]: A list of detection dictionaries.
    - list[dict[str, Any]]: A list of all targets.

    Notes
    -----
    - The function uses a progress bar (tqdm) to show the inference progress.
    - If 'num_samples' is not provided, the progress bar may not accurately
      reflect the total number of samples processed.
    - The batch_callback, if provided, is called after each batch is processed,
      allowing for real-time analysis or logging of results.
    """

    net.to(device, dtype=model_dtype)
    detections_list: list[dict[str, torch.Tensor]] = []
    target_list: list[dict[str, Any]] = []
    sample_paths: list[str] = []
    batch_size = dataloader.batch_size
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, targets, orig_sizes, masks, image_sizes in dataloader:
            # Inference
            inputs = inputs.to(device, dtype=model_dtype, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                detections = infer_batch(net, inputs, masks=masks, image_sizes=image_sizes, tta=tta)

            detections = InferenceTransform.postprocess(detections, image_sizes, orig_sizes)
            if targets[0] != settings.NO_LABEL:
                targets = InferenceTransform.postprocess(targets, image_sizes, orig_sizes)

            detections_list.extend(detections)

            # Set targets and sample list
            target_list.extend(targets)
            sample_paths.extend(file_paths)

            if batch_callback is not None:
                batch_callback(file_paths, inputs, detections, targets, image_sizes)

            # Update progress bar
            progress.update(n=batch_size)

    return (sample_paths, detections_list, target_list)
