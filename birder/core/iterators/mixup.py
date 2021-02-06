import numpy as np

from birder.core.iterators.label_smoothing import ImageRecordIterLabelSmoothing


class ImageRecordIterMixup(ImageRecordIterLabelSmoothing):
    """
    Mixup iterator

    Implement all classes with random pairs (AC + RP) mode with optional label smoothing.
    See paper: https://arxiv.org/abs/1710.09412 for more information.
    """

    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        smoothing_alpha: float = 0.1,
        mixup_beta: float = 0.2,
        **kwargs
    ):
        assert batch_size % 2 == 0, "batch size must be even for mixup"

        super().__init__(
            num_classes=num_classes, batch_size=batch_size, smoothing_alpha=smoothing_alpha, **kwargs
        )
        self.mixup_beta = mixup_beta

    def next(self):
        lam = np.random.beta(self.mixup_beta, self.mixup_beta)

        # Smooth labels
        batch = super().next()

        # Mixup data
        data = batch.data
        mixup_data = []
        for samples in data:
            mixup_samples = lam * samples + (1 - lam) * samples[::-1]
            mixup_data.append(mixup_samples)

        batch.data = mixup_data

        # Mixup labels
        labels = batch.label
        mixup_labels = []
        for label in labels:
            reversed_label = label[::-1]
            mixup_labels.append(lam * label + (1 - lam) * reversed_label)

        batch.label = mixup_labels

        return batch
