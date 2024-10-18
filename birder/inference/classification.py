from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.amp
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import five_crop
from tqdm import tqdm

from birder.results.classification import Results


def infer_image(
    net: torch.nn.Module | torch.ScriptModule,
    sample: Image.Image | str,
    transform: Callable[..., torch.Tensor],
    return_embedding: bool = False,
    tta: bool = False,
    device: Optional[torch.device] = None,
) -> tuple[npt.NDArray[np.float32], Optional[npt.NDArray[np.float32]]]:
    """
    Perform inference on a single image

    This convenience function allows for quick, one-off classification of an image.

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
    return infer_batch(net, input_tensor, return_embedding=return_embedding, tta=tta)


def infer_batch(
    net: torch.nn.Module | torch.ScriptModule, inputs: torch.Tensor, return_embedding: bool = False, tta: bool = False
) -> tuple[npt.NDArray[np.float32], Optional[npt.NDArray[np.float32]]]:
    if return_embedding is True:
        embedding_tensor: torch.Tensor = net.embedding(inputs)
        out: npt.NDArray[np.float32] = F.softmax(net.classify(embedding_tensor), dim=1).cpu().numpy()
        embedding: Optional[npt.NDArray[np.float32]] = embedding_tensor.cpu().numpy()

    elif tta is True:
        embedding = None
        (_, _, H, W) = inputs.size()
        crop_h = int(H * 0.8)
        crop_w = int(W * 0.8)
        tta_inputs = five_crop(inputs, size=[crop_h, crop_w])
        t = v2.Resize((H, W), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
        outs = []
        for tta_input in tta_inputs:
            outs.append(F.softmax(net(t(tta_input)), dim=1))

        out = torch.stack(outs).mean(axis=0).cpu().numpy()

    else:
        embedding = None
        out = F.softmax(net(inputs), dim=1).cpu().numpy()

    return (out, embedding)


def infer_dataloader(
    device: torch.device,
    net: torch.nn.Module | torch.ScriptModule,
    dataloader: DataLoader,
    return_embedding: bool = False,
    tta: bool = False,
    amp: bool = False,
    num_samples: Optional[int] = None,
    batch_callback: Optional[Callable[[list[str], npt.NDArray[np.float32], list[int]], None]] = None,
) -> tuple[list[str], npt.NDArray[np.float32], list[int], list[npt.NDArray[np.float32]]]:
    """
    Perform inference on a DataLoader using a given neural network.

    This function runs inference on a dataset provided through a DataLoader,
    optionally returning embeddings and using mixed precision (amp).

    Parameters
    ----------
    device
        The device to run the inference on.
    net
        The model to use for inference.
    dataloader
        The DataLoader containing the dataset to perform inference on.
    return_embedding
        Whether to return embeddings along with the outputs.
    tta
        Run inference with oversampling.
    amp
        Whether to use automatic mixed precision.
    num_samples
        The total number of samples in the dataloader.
    batch_callback
        A function to be called after each batch is processed. If provided, it
        should accept three arguments:
        - list[str]: A list of file paths for the current batch
        - npt.NDArray[np.float32]: The output array for the current batch
        - list[int]: A list of labels for the current batch

    Returns
    -------
        A tuple containing four elements:
        - list[str]: A list of all processed file paths.
        - npt.NDArray[np.float32]: A 2D numpy array of all outputs.
        - list[int]: A list of all labels.
        - list[npt.NDArray[np.float32]]: A list of embedding arrays if
          return_embedding is True, otherwise an empty list.

    Notes
    -----
    - The function uses a progress bar (tqdm) to show the inference progress.
    - If 'num_samples' is not provided, the progress bar may not accurately
      reflect the total number of samples processed.
    - The batch_callback, if provided, is called after each batch is processed,
      allowing for real-time analysis or logging of results.
    """

    net.to(device)
    embedding_list: list[npt.NDArray[np.float32]] = []
    out_list: list[npt.NDArray[np.float32]] = []
    labels: list[int] = []
    sample_paths: list[str] = []
    batch_size = dataloader.batch_size
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, targets in dataloader:
            # Inference
            inputs = inputs.to(device)

            with torch.amp.autocast(device.type, enabled=amp):
                (out, embedding) = infer_batch(net, inputs, return_embedding=return_embedding, tta=tta)

            out_list.append(out)
            if embedding is not None:
                embedding_list.append(embedding)

            # Set labels and sample list
            batch_labels = list(targets.cpu().numpy())
            labels.extend(batch_labels)
            sample_paths.extend(file_paths)

            if batch_callback is not None:
                batch_callback(file_paths, out, batch_labels)

            # Update progress bar
            progress.update(n=batch_size)

    outs = np.concatenate(out_list, axis=0)

    return (sample_paths, outs, labels, embedding_list)


def evaluate(
    device: torch.device,
    net: torch.nn.Module | torch.ScriptModule,
    dataloader: DataLoader,
    class_to_idx: dict[str, int],
    tta: bool = False,
    amp: bool = False,
    num_samples: Optional[int] = None,
) -> Results:
    (sample_paths, outs, labels, _) = infer_dataloader(
        device, net, dataloader, tta=tta, amp=amp, num_samples=num_samples
    )
    results = Results(sample_paths, labels, list(class_to_idx.keys()), outs)

    return results
