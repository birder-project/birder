from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.utils.data
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset.filters import default_collation_fn


def make_wds_loader(
    dataset: torch.utils.data.IterableDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    collate_fn: Optional[Callable[..., Any]],
    world_size: int,
    pin_memory: bool,
    partial: bool = True,
) -> DataLoader:
    if collate_fn is None:
        collate_fn = default_collation_fn

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=not partial,
    )

    if shuffle is True:
        # Shuffle and re-batch to mix samples from different workers
        dataloader = dataloader.unbatched().shuffle(1000).batched(batch_size, collation_fn=collate_fn, partial=partial)

    dataloader = dataloader.with_epoch(len(dataset) // (batch_size * world_size))

    return dataloader
