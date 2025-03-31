import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import torch.utils.data
import webdataset as wds
from torchvision.io import ImageReadMode
from torchvision.io import decode_image

from birder.common import fs_ops
from birder.common.training_utils import reduce_across_processes


def decode_sample_name(item: tuple[str, str, Any, int]) -> tuple[str, Any, int]:
    sample_name = item[0] + "/" + item[1]
    return (sample_name, item[2], item[3])


def wds_image_decoder(key: str, data: bytes) -> torch.Tensor:
    if key.endswith((".jpg", ".jpeg", ".webp", ".png")) is False:
        return None

    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return decode_image(tensor, mode=ImageReadMode.RGB)


def make_wds_dataset(
    wds_path: str | list[str],
    dataset_size: int,
    shuffle: bool,
    samples_names: bool,
    transform: Callable[..., torch.Tensor],
    *,
    cache_dir: Optional[str] = None,
) -> torch.utils.data.IterableDataset:
    if shuffle is True:
        shardshuffle = 100
    else:
        shardshuffle = False

    dataset = wds.WebDataset(
        wds_path, shardshuffle=shardshuffle, nodesplitter=wds.split_by_node, cache_dir=cache_dir, empty_check=False
    )
    if shuffle is True:
        dataset = dataset.shuffle(1000, initial=100)

    return_keys = ["jpeg;jpg;png;webp", "cls"]
    if samples_names is True:
        return_keys = ["__url__", "__key__"] + return_keys

    dataset = dataset.with_length(dataset_size, silent=True).decode("pil").to_tuple(*return_keys)
    # dataset = dataset.with_length(dataset_size).decode(wds_image_decoder).to_tuple(*return_keys)

    if samples_names is True:
        dataset = dataset.map(decode_sample_name)

    dataset = dataset.map(transform)

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

    size = reduce_across_processes(size, device, op=torch.distributed.ReduceOp.SUM)  # type: ignore

    return size


def prepare_wds_args(data_path: str, size: Optional[int], device: torch.device) -> tuple[str, int]:
    if "://" not in data_path:
        if ".." not in data_path:
            # Local path without braces, build brace argument
            (wds_path, _) = fs_ops.wds_braces_from_path(Path(data_path))

        else:
            wds_path = data_path

        if size is None:
            # If size not provided, we scan all tar files
            dataset_size = wds_size(wds_path, device)
        else:
            dataset_size = size

    else:
        # Remote path, we take the string as-is
        # Dataset size and must be provided
        assert size is not None
        wds_path = data_path
        dataset_size = size

    return (wds_path, dataset_size)


def wds_args_from_info(info_path: str, split: str) -> tuple[list[str], int]:
    info = fs_ops.read_wds_info(info_path)
    root = Path(info_path).parent

    size = info["splits"][split]["num_samples"]
    filenames = info["splits"][split]["filenames"]

    filenames = [os.path.join(root, f) for f in filenames]

    return (filenames, size)
