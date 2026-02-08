import logging
import unittest

import numpy as np
import torch
from sklearn.model_selection import RandomizedSearchCV

from birder.eval.methods.knn import evaluate_knn
from birder.eval.methods.mlp import evaluate_mlp
from birder.eval.methods.mlp import train_mlp
from birder.eval.methods.simpleshot import evaluate_simpleshot
from birder.eval.methods.svm import evaluate_svm
from birder.eval.methods.svm import train_svm

logging.disable(logging.CRITICAL)


def _make_separable_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    class0 = rng.normal(loc=-1.0, scale=0.1, size=(10, 2)).astype(np.float32)
    class1 = rng.normal(loc=1.0, scale=0.1, size=(10, 2)).astype(np.float32)

    train_features = np.vstack([class0[:8], class1[:8]])
    train_labels = np.array([0] * 8 + [1] * 8, dtype=np.int_)

    test_features = np.vstack([class0[8:], class1[8:]])
    test_labels = np.array([0] * 2 + [1] * 2, dtype=np.int_)

    return (train_features, train_labels, test_features, test_labels)


class TestMethods(unittest.TestCase):
    def test_train_svm_returns_search(self) -> None:
        train_features, train_labels, _, _ = _make_separable_data()
        svc = train_svm(train_features, train_labels, n_iter=2, n_jobs=1, seed=0)

        self.assertIsInstance(svc, RandomizedSearchCV)
        self.assertIsNotNone(svc.best_estimator_)
        self.assertIn("svc__C", svc.best_params_)

        pipeline = svc.best_estimator_
        self.assertEqual([name for name, _ in pipeline.steps], ["standardscaler", "svc"])

    def test_evaluate_svm_predictions(self) -> None:
        train_features, train_labels, test_features, test_labels = _make_separable_data()
        y_pred, y_true = evaluate_svm(
            train_features,
            train_labels,
            test_features,
            test_labels,
            n_iter=2,
            n_jobs=1,
            seed=0,
        )

        self.assertTrue(np.array_equal(y_true, test_labels))
        self.assertEqual(y_pred.shape, test_labels.shape)
        self.assertTrue(np.isin(y_pred, [0, 1]).all())

    def test_evaluate_simpleshot_predictions(self) -> None:
        train_features, train_labels, test_features, test_labels = _make_separable_data()
        y_pred, y_true = evaluate_simpleshot(train_features, train_labels, test_features, test_labels)

        self.assertTrue(np.array_equal(y_true, test_labels))
        self.assertEqual(y_pred.shape, test_labels.shape)
        self.assertTrue(np.isin(y_pred, [0, 1]).all())
        self.assertEqual(float(np.mean(y_pred == y_true)), 1.0)

    def test_evaluate_knn_predictions(self) -> None:
        train_features, train_labels, test_features, test_labels = _make_separable_data()
        y_pred, y_true = evaluate_knn(
            train_features, train_labels, test_features, test_labels, k=3, device=torch.device("cpu")
        )

        self.assertTrue(np.array_equal(y_true, test_labels))
        self.assertEqual(y_pred.shape, test_labels.shape)
        self.assertTrue(np.isin(y_pred, [0, 1]).all())
        self.assertEqual(float(np.mean(y_pred == y_true)), 1.0)

    def test_evaluate_knn_with_k_larger_than_train_set(self) -> None:
        train_features, train_labels, test_features, test_labels = _make_separable_data()
        y_pred, y_true = evaluate_knn(train_features, train_labels, test_features, test_labels, k=100)

        self.assertTrue(np.array_equal(y_true, test_labels))
        self.assertEqual(y_pred.shape, test_labels.shape)
        self.assertTrue(np.isin(y_pred, [0, 1]).all())

    def test_mlp_probe_predictions(self) -> None:
        rng = np.random.default_rng(0)

        # Create separable multi-label data
        n_train = 50
        n_test = 10
        n_features = 8
        n_classes = 2

        train_features = rng.normal(size=(n_train, n_features)).astype(np.float32)
        train_labels = (train_features[:, :n_classes] > 0).astype(np.float32)

        test_features = rng.normal(size=(n_test, n_features)).astype(np.float32)
        test_labels = (test_features[:, :n_classes] > 0).astype(np.float32)

        device = torch.device("cpu")
        model = train_mlp(
            train_features,
            train_labels,
            num_classes=n_classes,
            device=device,
            epochs=10,
            batch_size=8,
            hidden_dim=64,
            seed=0,
        )

        y_pred, y_true, macro_f1 = evaluate_mlp(model, test_features, test_labels, device=device)

        self.assertEqual(y_pred.shape, (n_test, n_classes))
        self.assertEqual(y_true.shape, (n_test, n_classes))
        self.assertTrue(np.isin(y_pred, [0, 1]).all())
        self.assertGreaterEqual(macro_f1, 0.0)
        self.assertLessEqual(macro_f1, 1.0)
