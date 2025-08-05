import logging
import math
from collections.abc import Callable
from typing import Any
from typing import Optional

import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def make_wds_loader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: Optional[int],
    collate_fn: Optional[Callable[..., Any]],
    world_size: int,
    pin_memory: bool,
    drop_last: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """
    NOTE: Validation in WDS is a bit messy, practically either some samples be seen twice, or skipped.
    """

    dataloader = wds.WebLoader(
        dataset,
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
        # drop_last actually does nothing here as the BatchSampler will always see a full batch
        epoch_size = math.floor(len(dataset) / (batch_size * world_size))
    else:
        epoch_size = math.ceil(len(dataset) / (batch_size * world_size))

    dataloader = dataloader.with_length(epoch_size, silent=True).repeat(2).with_epoch(epoch_size)

    return dataloader
