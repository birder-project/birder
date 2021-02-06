import unittest

import mxnet as mx

from birder.core.metrics import OneHotAccuracy
from birder.core.metrics import OneHotCrossEntropy
from birder.core.metrics import OneHotTopKAccuracy


class TestOneHotAccuracy(unittest.TestCase):
    def test_update(self):
        # 2 correct labels, 1 mistake
        predicts = [mx.nd.array([[0.3, 0.7], [0, 1.0], [0.4, 0.6]])]
        labels = [mx.nd.array([0, 1, 1])]
        acc = OneHotAccuracy()

        # Exception on "regular" encoded labels
        with self.assertRaises(mx.base.MXNetError):
            acc.update(labels, predicts)

        # Ensure one hot labels return correct accuracy
        labels = [labels[0].one_hot(2)]
        acc.update(labels, predicts)
        (_, score) = acc.get()
        self.assertAlmostEqual(0.6666666666, score)


class TestOneHotCrossEntropy(unittest.TestCase):
    def test_update(self):
        predicts = [mx.nd.array([[0, 1.0], [0, 1.0], [0.75, 0.25]])]
        labels = [mx.nd.array([1, 1, 0])]
        cross_entropy = OneHotCrossEntropy()

        # Exception on "regular" encoded labels
        with self.assertRaises(mx.base.MXNetError):
            cross_entropy.update(labels, predicts)

        # Ensure one hot labels return correct cross entropy
        labels = [labels[0].one_hot(2)]
        cross_entropy.update(labels, predicts)
        (_, score) = cross_entropy.get()
        self.assertAlmostEqual(0.095, score, places=2)


class TestOneHotTopKAccuracy(unittest.TestCase):
    def test_update(self):
        # 1 correct label, 2 top-2, 1 mistake
        predicts = [mx.nd.array([[0.3, 0.7, 0], [0, 1.0, 0], [0.4, 0.3, 0.2]])]
        labels = [mx.nd.array([0, 1, 2])]
        acc = OneHotTopKAccuracy(top_k=2)

        # Exception on "regular" encoded labels
        with self.assertRaises(mx.base.MXNetError):
            acc.update(labels, predicts)

        # Ensure one hot labels return correct top-k accuracy
        labels = [labels[0].one_hot(3)]
        acc.update(labels, predicts)
        (_, score) = acc.get()
        self.assertAlmostEqual(0.6666666666, score)
