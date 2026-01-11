import logging
import unittest

import torch
import webdataset as wds

from birder.data.dataloader.webdataset import make_wds_loader

logging.disable(logging.CRITICAL)


class TestWdsLoader(unittest.TestCase):
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
