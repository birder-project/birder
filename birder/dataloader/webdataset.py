from collections.abc import Callable
from typing import Any
from typing import Optional

import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset


def make_wds_loader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    collate_fn: Optional[Callable[..., Any]],
    world_size: int,
    pin_memory: bool,
    partial: bool = True,
) -> DataLoader:
    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=not partial,
    )
    dataloader = dataloader.with_epoch(len(dataset) // (batch_size * world_size))

    return dataloader
