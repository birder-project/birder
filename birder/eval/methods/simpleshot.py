"""
Paper "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning",
https://arxiv.org/abs/1911.04623
"""

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestCentroid

from birder.eval._embeddings import l2_normalize


def sample_k_shot(
    features: npt.NDArray[np.float32], labels: npt.NDArray[np.int_], k: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    """
    Sample k examples per class from the training set

    Returns
    -------
    sampled_features
        Features of shape (num_classes * k, embedding_dim).
    sampled_labels
        Labels of shape (num_classes * k,).
    """

    unique_labels = np.unique(labels)
    sampled_features_list: list[npt.NDArray[np.float32]] = []
    sampled_labels_list: list[npt.NDArray[np.int_]] = []

    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        class_labels = labels[mask]

        # Sample k examples (with replacement if fewer than k available)
        n_available = len(class_features)
        if n_available >= k:
            indices = rng.choice(n_available, size=k, replace=False)
        else:
            indices = rng.choice(n_available, size=k, replace=True)

        sampled_features_list.append(class_features[indices])
        sampled_labels_list.append(class_labels[indices])

    return (np.concatenate(sampled_features_list), np.concatenate(sampled_labels_list))


def normalize_features(
    train_features: npt.NDArray[np.float32], test_features: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Apply SimpleShot normalization: mean centering followed by L2 normalization
    """

    mean = train_features.mean(axis=0, keepdims=True)
    train_centered = train_features - mean
    test_centered = test_features - mean

    train_norm = l2_normalize(train_centered)
    test_norm = l2_normalize(test_centered)

    return (train_norm, test_norm)


def evaluate_simpleshot(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.int_],
    test_features: npt.NDArray[np.float32],
    test_labels: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Evaluate using SimpleShot method

    Parameters
    ----------
    train_features
        Training features of shape (n_train, embedding_dim).
    train_labels
        Training labels of shape (n_train,).
    test_features
        Test features of shape (n_test, embedding_dim).
    test_labels
        Test labels of shape (n_test,).

    Returns
    -------
    y_pred
        Predicted labels for test samples.
    y_true
        True labels for test samples (same as test_labels).
    """

    train_norm, test_norm = normalize_features(train_features, test_features)

    clf = NearestCentroid()
    clf.fit(train_norm, train_labels)
    y_pred = clf.predict(test_norm)

    return (y_pred, test_labels)
