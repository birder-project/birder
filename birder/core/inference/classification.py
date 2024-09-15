from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def infer_batch(
    net: torch.nn.Module | torch.ScriptModule, inputs: torch.Tensor, return_embedding: bool = False
) -> tuple[npt.NDArray[np.float32], Optional[npt.NDArray[np.float32]]]:
    if return_embedding is True:
        embedding_tensor: torch.Tensor = net.embedding(inputs)
        out: npt.NDArray[np.float32] = F.softmax(net.classify(embedding_tensor), dim=1).cpu().numpy()
        embedding: Optional[npt.NDArray[np.float32]] = embedding_tensor.cpu().numpy()

    else:
        embedding = None
        out = F.softmax(net(inputs), dim=1).cpu().numpy()

    return (out, embedding)


def infer_dataloader(
    device: torch.device,
    net: torch.nn.Module | torch.ScriptModule,
    dataloader: DataLoader,
    return_embedding: bool = False,
    amp: bool = False,
    num_samples: Optional[int] = None,
    batch_callback: Optional[Callable[[list[str], npt.NDArray[np.float32], list[int]], None]] = None,
) -> tuple[list[str], npt.NDArray[np.float32], list[int], list[npt.NDArray[np.float32]]]:
    embedding_list: list[npt.NDArray[np.float32]] = []
    out_list: list[npt.NDArray[np.float32]] = []
    labels: list[int] = []
    sample_paths: list[str] = []
    batch_size = dataloader.batch_size
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, targets in dataloader:
            # Predict
            inputs = inputs.to(device)

            with torch.amp.autocast(device.type, enabled=amp):
                (out, embedding) = infer_batch(net, inputs, return_embedding=return_embedding)

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
