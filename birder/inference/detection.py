from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.amp
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from birder.data.transforms.detection import InferenceTransform


def infer_image(
    net: torch.nn.Module | torch.ScriptModule,
    sample: Image.Image | str,
    transform: Callable[..., torch.Tensor],
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
    detections = infer_batch(net, input_tensor, **kwargs)
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
    **kwargs: Any,
) -> list[dict[str, torch.Tensor]]:
    (detections, _) = net(inputs, masks=masks, image_sizes=image_sizes, **kwargs)
    return detections  # type: ignore[no-any-return]


def infer_dataloader(
    device: torch.device,
    net: torch.nn.Module | torch.ScriptModule,
    dataloader: DataLoader,
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
                detections = infer_batch(net, inputs, masks, image_sizes)

            detections = InferenceTransform.postprocess(detections, image_sizes, orig_sizes)
            if targets[0] != -1:
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
