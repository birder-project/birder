from collections.abc import Sequence
from typing import Any

import torch

from birder.data.collators.naflex import NaFlexBatchProcessor


class NaFlexMultiScaleDataset(torch.utils.data.Dataset[tuple[Any, ...]]):
    """
    Apply one collator-selected NaFlex transform while loading each map-style batch

    DataLoader calls '__getitems__' with a batch of indices.
    The adapter keeps those indices lazy so each source image is transformed before the next one is decoded.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, processor: NaFlexBatchProcessor) -> None:
        self.dataset = dataset
        self.processor = processor

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return self.processor.transform_batch((self.dataset[index],))[0]

    def __getitems__(self, indices: Sequence[int]) -> list[tuple[Any, ...]]:
        samples = (self.dataset[index] for index in indices)
        return self.processor.transform_batch(samples)

    def __len__(self) -> int:
        return len(self.dataset)
