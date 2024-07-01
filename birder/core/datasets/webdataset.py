from collections.abc import Callable

import torch
import torch.utils.data
import webdataset as wds

from birder.common.training_utils import reduce_across_processes


def make_wds_dataset(
    wds_path: str, batch_size: int, dataset_size: int, transform: Callable[..., torch.Tensor]
) -> torch.utils.data.IterableDataset:
    dataset = (
        wds.WebDataset(wds_path, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(1000, initial=100)
        .with_length(dataset_size)
        .decode("pil")
        .to_tuple("jpeg;jpg;png;webp", "cls")
        .map(transform)
        .batched(batch_size, partial=True)
    )

    return dataset


def wds_size(wds_path: str, device: torch.device) -> int:
    dataset = wds.WebDataset(
        wds_path, select_files=lambda key_name: key_name.endswith("cls"), nodesplitter=wds.split_by_node
    ).batched(64, collation_fn=None, partial=True)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=8)
    size = 0
    for batch in dataloader:
        size += len(batch)

    size = reduce_across_processes(size, device)  # type: ignore

    return size
