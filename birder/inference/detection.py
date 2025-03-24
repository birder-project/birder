from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from birder.transforms.detection import batch_images


def infer_batch(net: torch.nn.Module | torch.ScriptModule, inputs: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    (detections, _) = net(inputs)
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
        Callable[[list[str], torch.Tensor, list[dict[str, torch.Tensor]], list[dict[str, Any]]], None]
    ] = None,
) -> tuple[list[str], list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """
    Perform inference on a DataLoader using a given neural network.

    This function runs inference on a dataset provided through a DataLoader,
    optionally using mixed precision (amp).

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
        for file_paths, inputs, targets in dataloader:
            # Inference
            inputs = [i.to(device, dtype=model_dtype) for i in inputs]
            inputs = batch_images(inputs)

            with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                detections = infer_batch(net, inputs)

            detections_list.extend(detections)

            # Set targets and sample list
            target_list.extend(targets)
            sample_paths.extend(file_paths)

            if batch_callback is not None:
                batch_callback(file_paths, inputs, detections, targets)

            # Update progress bar
            progress.update(n=batch_size)

    return (sample_paths, detections_list, target_list)
