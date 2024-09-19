from collections.abc import Callable
from typing import Any

import torch
import torch.utils.data
import webdataset as wds

from birder.common.training_utils import reduce_across_processes


def decode_sample_name(item: tuple[str, str, Any, int]) -> tuple[str, Any, int]:
    sample_name = item[0] + "/" + item[1]
    return (sample_name, item[2], item[3])


def make_wds_dataset(
    wds_path: str,
    batch_size: int,
    dataset_size: int,
    shuffle: bool,
    samples_names: bool,
    transform: Callable[..., torch.Tensor],
) -> torch.utils.data.IterableDataset:
    dataset = wds.WebDataset(wds_path, shardshuffle=shuffle, nodesplitter=wds.split_by_node, empty_check=False)
    if shuffle is True:
        dataset = dataset.shuffle(1000, initial=100)

    return_keys = ["jpeg;jpg;png;webp", "cls"]
    if samples_names is True:
        return_keys = ["__url__", "__key__"] + return_keys

    dataset = dataset.with_length(dataset_size).decode("pil").to_tuple(*return_keys)

    if samples_names is True:
        dataset = dataset.map(decode_sample_name)

    dataset = dataset.map(transform).batched(batch_size, partial=True)

    return dataset


def wds_size(wds_path: str, device: torch.device) -> int:
    dataset = wds.WebDataset(
        wds_path,
        shardshuffle=False,
        select_files=lambda key_name: key_name.endswith("cls"),
        nodesplitter=wds.split_by_node,
        empty_check=False,
    ).batched(64, collation_fn=None, partial=True)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=8)
    size = 0
    for batch in dataloader:
        size += len(batch)

    size = reduce_across_processes(size, device)  # type: ignore

    return size
