import logging
import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
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
    persistent_workers: bool = False,
    shuffle: bool = False,
    *,
    exact: bool = False,
    infinite: bool = False,
    batcher: Optional[Callable[[Iterable[tuple[Any, ...]]], Iterator[tuple[Any, ...]]]] = None,
    num_shards: Optional[int] = None,
) -> DataLoader:
    assert exact is False or infinite is False

    if (batcher is None) != (num_shards is None):
        raise ValueError("batcher and num_shards must be provided together")
    if batcher is not None and num_shards is not None:
        if num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {num_shards}")

        max_active_workers = num_shards // world_size
        if max_active_workers == 0:
            raise ValueError("Custom WDS batching requires at least one shard per distributed rank")
        if num_workers > 0 and num_workers > max_active_workers:
            logger.warning(
                f"Reducing WDS workers from {num_workers} to {max_active_workers} to keep custom batches aligned "
                "across distributed ranks"
            )
            num_workers = max_active_workers

    if infinite is True:
        dataset_iterable = dataset.repeat()
    elif exact is False:
        dataset_iterable = dataset.repeat()
    else:
        dataset_iterable = dataset

    web_loader_batch_size: Optional[int] = batch_size
    web_loader_collate_fn = collate_fn
    web_loader_drop_last = drop_last
    if batcher is not None:
        # DataPipeline.repeat() repeats the complete pipeline, including stages composed after it. Wrap the repeated
        # dataset so the custom batcher sees one continuous stream instead of a separate tail for every repetition.
        dataset_iterable = wds.DataPipeline(dataset_iterable, batcher)
        web_loader_batch_size = None
        web_loader_collate_fn = None
        web_loader_drop_last = False  # The custom batcher owns the batch boundary and handles drop_last itself

    dataloader = wds.WebLoader(
        dataset_iterable,
        batch_size=web_loader_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=web_loader_collate_fn,
        pin_memory=pin_memory,
        drop_last=web_loader_drop_last,
        persistent_workers=persistent_workers,
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
    if exact is False and infinite is False:
        dataloader = dataloader.with_epoch(epoch_size)

    return dataloader
