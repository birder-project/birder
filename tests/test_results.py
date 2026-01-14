import logging
import unittest

import numpy as np
import torch

from birder.conf import settings
from birder.results.classification import Results
from birder.results.classification import SparseResults
from birder.results.classification import top_k_accuracy_score
from birder.results.detection import Results as DetectionResults

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
        output = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )

        results = Results(sample_list, labels, label_names, output)
        self.assertFalse(results.missing_labels)
        self.assertAlmostEqual(results.accuracy, 1.0 / 6.0)
        self.assertAlmostEqual(results.top_k, 5.0 / 6.0)
        self.assertEqual(results._top_k_indices, [1, 2, 3, 4, 5])  # pylint: disable=protected-access
        self.assertEqual(results.predictions.tolist(), [2, 1, 2, 2, 2, 2])
        self.assertEqual(results.prediction_names.to_list(), ["l2", "l1", "l2", "l2", "l2", "l2"])

        report = results.detailed_report()
        self.assertSequenceEqual(report["Class"].to_list(), [0, 1, 2, 3])

        # Test confusion matrix
        cnf = results.confusion_matrix
        self.assertSequenceEqual(cnf.shape, (4, 4))
        expected = np.array([[0, 1, 1, 0], [0, 0, 2, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.int_)
        self.assertSequenceEqual(cnf.tolist(), expected.tolist())

    def test_results_filter(self) -> None:
        sample_list = ["file1.jpeg", "file2.jpg", "file3.jpeg", "file4.jpeg", "file5.png", "file6.webp"]
        labels = [0, 0, 2, 1, 1, 3]
        label_names = ["l0", "l1", "l2", "l3"]
        output = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )

        results = Results(sample_list, labels, label_names, output)
        filtered_results = results.filter_by_labels([0, 2])

        self.assertEqual(len(filtered_results), 3)
        self.assertFalse(filtered_results.missing_labels)
        self.assertAlmostEqual(filtered_results.accuracy, 1.0 / 3.0)

    def test_most_confused_pairs(self) -> None:
        sample_list = [f"file{i}.jpg" for i in range(9)]
        labels = [0, 0, 0, 0, 1, 1, 1, 2, 2]
        predictions = [1, 1, 1, 0, 1, 1, 0, 2, 1]
        label_names = ["l0", "l1", "l2"]
        output = np.zeros((len(sample_list), len(label_names)), dtype=np.float32)
        for idx, pred in enumerate(predictions):
            output[idx, pred] = 1.0

        results = Results(sample_list, labels, label_names, output)
        pairs = results.most_confused_pairs(n=1)

        self.assertEqual(pairs.height, 1)
        self.assertSequenceEqual(pairs["predicted"].to_list(), ["l1"])
        self.assertSequenceEqual(pairs["actual"].to_list(), ["l0"])
        self.assertSequenceEqual(pairs["amount"].to_list(), [3])
        self.assertSequenceEqual(pairs["reverse"].to_list(), [1])

    def test_partial_results(self) -> None:
        sample_list = ["file1.jpeg", "file2.jpg", "file3.jpeg", "file4.jpeg", "file5.png", "file6.webp"]
        labels = [0, settings.NO_LABEL, 2, settings.NO_LABEL, 1, 3]
        label_names = ["l0", "l1", "l2", "l3"]
        output = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )

        results = Results(sample_list, labels, label_names, output)

        self.assertTrue(results.missing_labels)
        self.assertFalse(results.missing_all_labels)
        self.assertEqual(results._valid_length, 4)  # pylint: disable=protected-access
        self.assertAlmostEqual(results.accuracy, 1.0 / 4.0)
        self.assertAlmostEqual(results.top_k, 3.0 / 4.0)
        self.assertEqual(results._top_k_indices, [2, 4, 5])  # pylint: disable=protected-access
        self.assertEqual(results.predictions.tolist(), [2, 1, 2, 2, 2, 2])
        self.assertEqual(results.prediction_names.to_list(), ["l2", "l1", "l2", "l2", "l2", "l2"])

        report = results.detailed_report()
        self.assertSequenceEqual(report["Class"].to_list(), [0, 1, 2, 3])

    def test_sparse_results(self) -> None:
        sample_list = ["file1.jpeg", "file2.jpg", "file3.jpeg", "file4.jpeg", "file5.png", "file6.webp"]
        labels = [0, 0, 2, 1, 1, 3]
        label_names = ["l0", "l1", "l2", "l3"]
        output = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )

        results = SparseResults(sample_list, labels, label_names, output, sparse_k=3)
        self.assertFalse(results.missing_labels)
        self.assertAlmostEqual(results.accuracy, 1.0 / 6.0)
        self.assertAlmostEqual(results.top_k, 5.0 / 6.0)
        self.assertEqual(results._top_k_indices, [1, 2, 3, 4, 5])  # pylint: disable=protected-access
        self.assertEqual(results.predictions.tolist(), [2, 1, 2, 2, 2, 2])
        self.assertEqual(results.prediction_names.to_list(), ["l2", "l1", "l2", "l2", "l2", "l2"])

        report = results.detailed_report()
        self.assertSequenceEqual(report["Class"].to_list(), [0, 1, 2, 3])

        cnf = results.confusion_matrix
        self.assertSequenceEqual(cnf.shape, (4, 4))

    def test_sparse_results_filter(self) -> None:
        sample_list = ["file1.jpeg", "file2.jpg", "file3.jpeg", "file4.jpeg", "file5.png", "file6.webp"]
        labels = [0, 0, 2, 1, 1, 3]
        label_names = ["l0", "l1", "l2", "l3"]
        output = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )

        results = SparseResults(sample_list, labels, label_names, output, sparse_k=3)
        filtered_results = results.filter_by_labels([0, 2])

        self.assertEqual(len(filtered_results), 3)
        self.assertFalse(filtered_results.missing_labels)
        self.assertAlmostEqual(filtered_results.accuracy, 1.0 / 3.0)

    def test_partial_sparse_results(self) -> None:
        sample_list = ["file1.jpeg", "file2.jpg", "file3.jpeg", "file4.jpeg", "file5.png", "file6.webp"]
        labels = [0, settings.NO_LABEL, 2, settings.NO_LABEL, 1, 3]
        label_names = ["l0", "l1", "l2", "l3"]
        output = np.array(
            [
                [0.1, 0.25, 0.5, 0.15],
                [0.25, 0.5, 0.15, 0.1],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.25, 0.5, 0.15],
                [0.1, 0.15, 0.5, 0.25],
                [0.1, 0.25, 0.5, 0.15],
            ]
        )

        results = SparseResults(sample_list, labels, label_names, output, sparse_k=3)

        self.assertTrue(results.missing_labels)
        self.assertFalse(results.missing_all_labels)
        self.assertEqual(results._valid_length, 4)  # pylint: disable=protected-access
        self.assertAlmostEqual(results.accuracy, 1.0 / 4.0)
        self.assertAlmostEqual(results.top_k, 3.0 / 4.0)
        self.assertEqual(results._top_k_indices, [2, 4, 5])  # pylint: disable=protected-access
        self.assertEqual(results.predictions.tolist(), [2, 1, 2, 2, 2, 2])
        self.assertEqual(results.prediction_names.to_list(), ["l2", "l1", "l2", "l2", "l2", "l2"])

        report = results.detailed_report()
        self.assertSequenceEqual(report["Class"].to_list(), [0, 1, 2, 3])


class TestDetectionResults(unittest.TestCase):
    def test_results(self) -> None:
        sample_paths = ["img1.jpg", "img2.jpg"]
        class_to_idx = {"cat": 1, "dog": 2}
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]), "labels": torch.tensor([2])},
        ]
        detections = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([1]),
            },
            {
                "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([2]),
            },
        ]

        results = DetectionResults(sample_paths, targets, detections, class_to_idx)
        self.assertEqual(len(results), 2)
        self.assertEqual(results.label_names, ["Background", "cat", "dog"])
        self.assertAlmostEqual(results.map, 1.0, places=4)

        report = results.detailed_report()
        self.assertSequenceEqual(report["Class"].to_list(), [1, 2])
        self.assertSequenceEqual(report["Objects"].to_list(), [1, 1])

    def test_confusion_matrix(self) -> None:
        sample_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        class_to_idx = {"cat": 1, "dog": 2}
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), "labels": torch.tensor([1, 2])},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([2])},
        ]
        detections = [
            {
                "boxes": torch.tensor(
                    [
                        [0.0, 0.0, 10.0, 10.0],
                        [20.0, 20.0, 30.0, 30.0],
                        [40.0, 40.0, 50.0, 50.0],
                    ]
                ),
                "scores": torch.tensor([0.95, 0.9, 0.4]),
                "labels": torch.tensor([1, 2, 1]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [0.0, 0.0, 10.0, 10.0],
                        [0.0, 0.0, 9.0, 9.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "scores": torch.tensor([0.9, 0.8, 0.6]),
                "labels": torch.tensor([2, 1, 2]),
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.4]),
                "labels": torch.tensor([2]),
            },
        ]

        results = DetectionResults(sample_paths, targets, detections, class_to_idx)
        cnf = results.confusion_matrix(score_threshold=0.5, iou_threshold=0.5)
        expected = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=np.int_)
        self.assertSequenceEqual(cnf.tolist(), expected.tolist())

    def test_confusion_matrix_iou_threshold(self) -> None:
        sample_paths = ["img1.jpg"]
        class_to_idx = {"cat": 1}
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]
        detections = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]

        results = DetectionResults(sample_paths, targets, detections, class_to_idx)
        cnf_low = results.confusion_matrix(score_threshold=0.5, iou_threshold=0.5)
        cnf_high = results.confusion_matrix(score_threshold=0.5, iou_threshold=0.7)

        expected_low = np.array([[0, 0], [0, 1]], dtype=np.int_)
        expected_high = np.array([[0, 1], [1, 0]], dtype=np.int_)
        self.assertSequenceEqual(cnf_low.tolist(), expected_low.tolist())
        self.assertSequenceEqual(cnf_high.tolist(), expected_high.tolist())

    def test_most_confused_pairs(self) -> None:
        sample_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]
        class_to_idx = {"cat": 1, "dog": 2}
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[20.0, 20.0, 30.0, 30.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[40.0, 40.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[0.0, 20.0, 10.0, 30.0]]), "labels": torch.tensor([2])},
            {"boxes": torch.tensor([[20.0, 0.0, 30.0, 10.0]]), "labels": torch.tensor([2])},
            {
                "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
                "labels": torch.tensor([], dtype=torch.int64),
            },
        ]
        detections = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([2]),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([2]),
            },
            {
                "boxes": torch.tensor([[40.0, 40.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.85]),
                "labels": torch.tensor([2]),
            },
            {
                "boxes": torch.tensor([[0.0, 20.0, 10.0, 30.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            },
            {
                "boxes": torch.tensor([[20.0, 0.0, 30.0, 10.0]]),
                "scores": torch.tensor([0.85]),
                "labels": torch.tensor([1]),
            },
            {
                "boxes": torch.tensor([[60.0, 60.0, 70.0, 70.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([2]),
            },
        ]

        results = DetectionResults(sample_paths, targets, detections, class_to_idx)
        pairs = results.most_confused_pairs(n=3, score_threshold=0.5, iou_threshold=0.5)
        self.assertEqual(pairs.height, 3)
        self.assertSequenceEqual(pairs["predicted"].to_list(), ["dog", "cat", "dog"])
        self.assertSequenceEqual(pairs["actual"].to_list(), ["cat", "dog", "Background"])
        self.assertSequenceEqual(pairs["amount"].to_list(), [3, 2, 1])
        self.assertSequenceEqual(pairs["reverse"].to_list(), [2, 3, 0])
