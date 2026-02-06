"""
K-Nearest Neighbors classifier for few-shot learning evaluation

Uses cosine similarity (dot product of L2-normalized features) with temperature-scaled softmax voting.
"""

import numpy as np
import numpy.typing as npt

from birder.eval._embeddings import l2_normalize


def evaluate_knn(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.int_],
    test_features: npt.NDArray[np.float32],
    test_labels: npt.NDArray[np.int_],
    k: int,
    temperature: float = 0.07,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Evaluate using K-Nearest Neighbors with cosine similarity and soft voting

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
    k
        Number of nearest neighbors.
    temperature
        Temperature for softmax scaling.

    Returns
    -------
    y_pred
        Predicted labels for test samples.
    y_true
        True labels for test samples (same as test_labels).
    """

    # Cosine similarity
    train_norm = l2_normalize(train_features)
    test_norm = l2_normalize(test_features)
    similarities = test_norm @ train_norm.T

    # Get top-k neighbors
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    top_k_sims = np.take_along_axis(similarities, top_k_indices, axis=1)
    top_k_labels = train_labels[top_k_indices]

    # Temperature-scaled softmax voting
    top_k_sims_scaled = top_k_sims / temperature
    top_k_sims_scaled = top_k_sims_scaled - top_k_sims_scaled.max(axis=1, keepdims=True)
    weights = np.exp(top_k_sims_scaled)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Weighted voting
    num_classes = train_labels.max() + 1
    votes = np.zeros((len(test_features), num_classes), dtype=np.float32)
    for i in range(k):
        np.add.at(votes, (np.arange(len(test_features)), top_k_labels[:, i]), weights[:, i])

    y_pred = votes.argmax(axis=1).astype(np.int_)

    return (y_pred, test_labels)
