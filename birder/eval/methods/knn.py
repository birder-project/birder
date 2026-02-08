"""
K-Nearest Neighbors classifier for few-shot learning evaluation

Uses cosine similarity (dot product of L2-normalized features) with temperature-scaled softmax voting.
"""

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


def evaluate_knn(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.int_],
    test_features: npt.NDArray[np.float32],
    test_labels: npt.NDArray[np.int_],
    k: int,
    temperature: float = 0.07,
    device: torch.device = torch.device("cpu"),
    chunk_size: int = 4096,
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
    device
        Device to run on.
    chunk_size
        Number of test samples to process per chunk.

    Returns
    -------
    y_pred
        Predicted labels for test samples.
    y_true
        True labels for test samples (same as test_labels).
    """

    effective_k = min(k, train_features.shape[0])

    train_tensor = torch.from_numpy(train_features).to(device=device, dtype=torch.float32)
    train_norm = F.normalize(train_tensor, p=2.0, dim=1, eps=1e-12)
    train_labels_tensor = torch.from_numpy(train_labels.astype(np.int64, copy=False)).to(
        device=device, dtype=torch.long
    )

    num_classes = train_labels_tensor.max().item() + 1
    pred_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(test_features), chunk_size):
            stop = min(start + chunk_size, len(test_features))
            test_chunk = torch.from_numpy(test_features[start:stop]).to(device=device, dtype=torch.float32)
            test_norm = F.normalize(test_chunk, p=2.0, dim=1, eps=1e-12)

            similarities = test_norm @ train_norm.T
            top_k_sims, top_k_indices = torch.topk(similarities, k=effective_k, dim=1, largest=True, sorted=False)
            top_k_labels = train_labels_tensor[top_k_indices]

            weights = torch.softmax(top_k_sims / temperature, dim=1)
            votes = torch.zeros((test_chunk.size(0), num_classes), dtype=torch.float32, device=device)
            votes.scatter_add_(1, top_k_labels, weights)

            pred_chunks.append(votes.argmax(dim=1))

    if len(pred_chunks) == 0:
        y_pred = np.empty((0,), dtype=np.int_)
    else:
        y_pred = torch.concat(pred_chunks, dim=0).cpu().numpy().astype(np.int_, copy=False)

    return (y_pred, test_labels)
