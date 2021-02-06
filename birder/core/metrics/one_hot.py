import mxnet as mx


class OneHotCrossEntropy(mx.metric.CrossEntropy):
    def update(self, labels, preds):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(label.argmax(axis=1))

        super().update(encoded_labels, preds)


class OneHotAccuracy(mx.metric.Accuracy):
    def update(self, labels, preds):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(label.argmax(axis=1))

        super().update(encoded_labels, preds)


class OneHotTopKAccuracy(mx.metric.TopKAccuracy):
    def update(self, labels, preds):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(label.argmax(axis=1))

        super().update(encoded_labels, preds)
