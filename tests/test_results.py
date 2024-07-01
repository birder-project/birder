import logging
import unittest

import numpy as np

from birder.core.results.classification import Results
from birder.core.results.classification import top_k_accuracy_score

logging.disable(logging.CRITICAL)


class TestClassification(unittest.TestCase):
    def test_top_k_accuracy_score(self) -> None:
        y_true = np.array([0, 0, 2, 1, 1, 3])
        y_pred = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )
        indices = top_k_accuracy_score(y_true, y_pred, top_k=2)
        self.assertEqual(indices, [1, 2, 3])

    def test_results(self) -> None:
        sample_list = ["file1.jpeg", "file2.jpg", "file3.jpeg", "file4.jpeg", "file5.png", "file6.webp"]
        labels = [0, 0, 2, 1, 1, 3]
        label_names = ["l0", "l1", "l2", "l3"]
        output = [
            np.array([0.1, 0.25, 0.5, 0.15]),
            np.array([0.25, 0.5, 0.15, 0.1]),
            np.array([0.1, 0.25, 0.5, 0.15]),
            np.array([0.1, 0.25, 0.5, 0.15]),
            np.array([0.1, 0.15, 0.5, 0.25]),
            np.array([0.1, 0.25, 0.5, 0.15]),
        ]

        results = Results(sample_list, labels, label_names, output)
        self.assertAlmostEqual(results.accuracy, 1.0 / 6.0)
        self.assertAlmostEqual(results.top_k, 5.0 / 6.0)
        self.assertEqual(results.predictions.tolist(), [2, 1, 2, 2, 2, 2])
        self.assertEqual(results.prediction_names.tolist(), ["l2", "l1", "l2", "l2", "l2", "l2"])
        self.assertFalse(results.missing_labels)

        report = results.detailed_report()
        self.assertSequenceEqual(report["Class"].to_list(), [0, 1, 2, 3])

        cnf = results.confusion_matrix
        self.assertSequenceEqual(cnf.shape, (4, 4))
