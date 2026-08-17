import logging
import unittest
from collections.abc import Iterable
from collections.abc import Iterator
from functools import partial

import torch
import webdataset as wds

from birder.data.collators.naflex import NaFlexBatchProcessor
from birder.data.collators.naflex import NaFlexTrainingCollator
from birder.data.dataloader.webdataset import make_wds_loader

logging.disable(logging.CRITICAL)

Sample = tuple[torch.Tensor, torch.Tensor]


def _mock_dataset(length: int = 10) -> wds.DataPipeline:
    sample = (torch.ones(1), torch.tensor(0))
    return wds.DataPipeline(wds.MockDataset(sample=sample, length=length)).with_length(length, silent=True)


def _unused_batcher(source: Iterable[Sample]) -> Iterator[Sample]:
    """Minimal custom batcher for loader configuration tests"""

    del source
    return iter(())


class _WorkerSplitMockDataset(torch.utils.data.IterableDataset):  # pylint: disable=abstract-method
    def __init__(self, sample: Sample, length: int) -> None:
        self.sample = sample
        self.length = length

    def __iter__(self) -> Iterator[Sample]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            step = 1
        else:
            start = worker_info.id
            step = worker_info.num_workers

        for _ in range(start, self.length, step):
            yield self.sample

    def __len__(self) -> int:
        return self.length


class TestWdsLoader(unittest.TestCase):
    def test_custom_batcher_requires_shard_metadata(self) -> None:
        with self.assertRaises(ValueError):
            # Batcher and num_shards must be provided together
            make_wds_loader(
                dataset=_mock_dataset(),
                batch_size=2,
                num_workers=0,
                prefetch_factor=None,
                collate_fn=None,
                world_size=1,
                pin_memory=False,
                batcher=_unused_batcher,
            )

    def test_custom_batcher_limits_workers_to_shards_per_rank(self) -> None:
        loader = make_wds_loader(
            dataset=_mock_dataset(),
            batch_size=2,
            num_workers=4,
            prefetch_factor=1,
            collate_fn=None,
            world_size=2,
            pin_memory=False,
            batcher=_unused_batcher,
            num_shards=5,
        )

        torch_loader = loader.pipeline[0]
        self.assertEqual(torch_loader.num_workers, 2)

    def test_custom_batcher_requires_one_shard_per_rank(self) -> None:
        with self.assertRaises(ValueError):
            # At least one shard per distributed rank
            make_wds_loader(
                dataset=_mock_dataset(),
                batch_size=2,
                num_workers=0,
                prefetch_factor=None,
                collate_fn=None,
                world_size=2,
                pin_memory=False,
                batcher=_unused_batcher,
                num_shards=1,
            )

    def test_custom_batcher_produces_full_batches_with_multiple_workers(self) -> None:
        n_samples = 24
        batch_size = 5
        sample = (torch.ones((1, 4, 4)), torch.tensor(1))
        mock_ds = _WorkerSplitMockDataset(sample, n_samples)
        processor = NaFlexBatchProcessor(
            NaFlexTrainingCollator(2),
            {4: torch.nn.Identity()},
            seed=0,
        )
        dataset = wds.DataPipeline(mock_ds).with_length(n_samples, silent=True)

        loader = make_wds_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=2,
            prefetch_factor=1,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
            batcher=partial(processor.iter_batches, batch_size=batch_size, drop_last=False),
            num_shards=2,
        )

        batches = list(loader)
        self.assertEqual(len(batches), 5)
        for inputs, _targets in batches:
            patches, _grid_sizes, _valid_mask = inputs
            self.assertEqual(patches.size(0), batch_size)

    def test_wds_loader_infinite_mode(self) -> None:
        n_samples = 50
        batch_size = 10

        # Create synthetic data: yields (tensor, label)
        mock_ds = wds.MockDataset(sample=(torch.randn(5), torch.tensor(1)), length=n_samples)
        dataset = wds.DataPipeline(mock_ds).with_length(n_samples, silent=True)

        # Infinite dataloader
        loader = make_wds_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            prefetch_factor=None,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            exact=False,
            infinite=True,
        )

        self.assertEqual(len(loader), 5)

        iterator = iter(loader)
        batches_fetched = 0
        try:
            for _ in range(15):
                next(iterator)
                batches_fetched += 1
        except StopIteration:
            self.fail("Infinite loader stopped iterating prematurely")

        self.assertEqual(batches_fetched, 15)

    def test_wds_loader_exact_mode(self) -> None:
        n_samples = 23
        batch_size = 10

        # Create synthetic data: yields (tensor, label)
        mock_ds = wds.MockDataset(sample=(torch.randn(5), torch.tensor(1)), length=n_samples)
        dataset = wds.DataPipeline(mock_ds).with_length(n_samples, silent=True)

        # Exact dataloader, partial batch
        loader = make_wds_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            prefetch_factor=None,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            exact=True,
            infinite=False,
        )

        self.assertEqual(len(loader), 3)

        batches = list(loader)
        self.assertEqual(len(batches), 3)

        # Check partial batch size
        self.assertEqual(len(batches[-1][0]), 3)

        # Exact dataloader, drop last
        loader = make_wds_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            prefetch_factor=None,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            exact=True,
            infinite=False,
        )

        self.assertEqual(len(loader), 2)

        batches = list(loader)
        self.assertEqual(len(batches), 2)

        # Check last batch
        self.assertEqual(len(batches[-1][0]), batch_size)
