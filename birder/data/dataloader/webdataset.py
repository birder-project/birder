import logging
import math
from collections.abc import Callable
from typing import Any
from typing import Optional

import webdataset as wds
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def make_wds_loader(
    dataset: wds.WebDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: Optional[int],
    collate_fn: Optional[Callable[..., Any]],
    world_size: int,
    pin_memory: bool,
    drop_last: bool = False,
    shuffle: bool = False,
    *,
    exact: bool = False,
) -> DataLoader:
    dataloader = wds.WebLoader(
        dataset.repeat() if exact is False else dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if shuffle is True:
        logger.info("WDS extra shuffle enabled: applying global batch-level shuffling")
        dataloader = dataloader.unbatched().shuffle(1000).batched(batch_size)

    dataloader.batch_size = batch_size
    if drop_last is True:
        epoch_size = math.floor(len(dataset) / (batch_size * world_size))
    else:
        epoch_size = math.ceil(len(dataset) / (batch_size * world_size))

    dataloader = dataloader.with_length(epoch_size, silent=True)
    if exact is False:
        dataloader = dataloader.with_epoch(epoch_size)

    return dataloader
