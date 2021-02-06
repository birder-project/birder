from birder.core.iterators.base import ImageRecordIterOneHot


class ImageRecordIterLabelSmoothing(ImageRecordIterOneHot):
    """
    Label Smoothing

    See paper: https://arxiv.org/abs/1512.00567 (chapter 7).
    """

    def __init__(self, num_classes: int, batch_size: int, smoothing_alpha: float = 0.1, **kwargs):
        super().__init__(num_classes=num_classes, batch_size=batch_size, **kwargs)
        self.num_classes = num_classes
        self.smoothing_alpha = smoothing_alpha

    def next(self):
        on_value = 1 - self.smoothing_alpha + (self.smoothing_alpha / self.num_classes)
        off_value = self.smoothing_alpha / self.num_classes

        batch = self.iter.next()
        labels = batch.label
        smoothed_labels = []
        for label in labels:
            smoothed_labels.append(label.one_hot(self.num_classes, on_value=on_value, off_value=off_value))

        batch.label = smoothed_labels

        return batch
