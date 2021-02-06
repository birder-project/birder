import mxnet as mx


class ImageRecordIterWrapper(mx.io.DataIter):
    """
    TODO
    """

    def __init__(self, batch_size: int, **kwargs):
        super().__init__(batch_size=batch_size)
        self.iter = mx.io.ImageRecordIter(batch_size=batch_size, **kwargs)

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()

    @property
    def provide_data(self):
        return self.iter.provide_data

    @property
    def provide_label(self):
        return self.iter.provide_label

    def next(self):
        return self.iter.next()


class ImageRecordIterOneHot(ImageRecordIterWrapper):
    """
    TODO
    """

    def __init__(self, num_classes: int, batch_size: int, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.num_classes = num_classes

    @property
    def provide_label(self):
        provide_label = self.iter.provide_label
        labels = []
        for data_desc in provide_label:
            one_hot_data_desc = mx.io.DataDesc(
                data_desc.name, data_desc.shape + (self.num_classes,), data_desc.dtype
            )
            labels.append(one_hot_data_desc)

        return labels

    def next(self):
        batch = self.iter.next()
        labels = batch.label
        one_hot_labels = []
        for label in labels:
            one_hot_labels.append(label.one_hot(self.num_classes))

        batch.label = one_hot_labels

        return batch
