"""
Embedding retrieval evaluation with mAP and Recall@K metrics
"""

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


def evaluate_retrieval(
    gallery_features: npt.NDArray[np.float32],
    gallery_labels: npt.NDArray[np.int_],
    query_features: npt.NDArray[np.float32],
    query_labels: npt.NDArray[np.int_],
    k_values: list[int],
    device: torch.device = torch.device("cpu"),
    chunk_size: int = 1024,
) -> tuple[float, dict[int, float]]:
    """
    Evaluate retrieval using cosine similarity over L2-normalized embeddings

    Parameters
    ----------
    gallery_features
        Gallery features of shape (n_gallery, embedding_dim).
    gallery_labels
        Gallery labels of shape (n_gallery,).
    query_features
        Query features of shape (n_query, embedding_dim).
    query_labels
        Query labels of shape (n_query,).
    k_values
        Recall@K values to report.
    device
        Device to run on.
    chunk_size
        Number of query samples to process per chunk.

    Returns
    -------
    mean_average_precision
        Mean average precision across all queries.
    recall_at_k
        Recall@K values keyed by K.
    """

    max_k = min(max(k_values), len(gallery_features))
    gallery_tensor = torch.from_numpy(gallery_features).to(device=device, dtype=torch.float32)
    gallery_norm = F.normalize(gallery_tensor, p=2.0, dim=1, eps=1e-12)
    gallery_labels_tensor = torch.from_numpy(gallery_labels.astype(np.int64, copy=False)).to(
        device=device, dtype=torch.long
    )

    ap_chunks: list[torch.Tensor] = []
    recall_hits: dict[int, int] = {k: 0 for k in k_values}
    with torch.inference_mode():
        for start in range(0, len(query_features), chunk_size):
            stop = min(start + chunk_size, len(query_features))
            query_chunk = torch.from_numpy(query_features[start:stop]).to(device=device, dtype=torch.float32)
            query_norm = F.normalize(query_chunk, p=2.0, dim=1, eps=1e-12)
            query_labels_tensor = torch.from_numpy(query_labels[start:stop].astype(np.int64, copy=False)).to(
                device=device, dtype=torch.long
            )

            similarities = query_norm @ gallery_norm.T
            relevant = query_labels_tensor[:, None] == gallery_labels_tensor[None, :]
            num_positives = relevant.sum(dim=1)
            if torch.any(num_positives == 0):
                raise RuntimeError("Found queries without any positive matches in the gallery")

            sorted_indices = similarities.argsort(dim=1, descending=True)
            sorted_relevant = relevant.gather(1, sorted_indices)

            cumulative_hits = sorted_relevant.cumsum(dim=1, dtype=torch.int64).to(dtype=torch.float32)
            ranks = torch.arange(1, sorted_relevant.shape[1] + 1, device=device, dtype=torch.float32)
            precision_at_rank = cumulative_hits / ranks
            average_precision = (precision_at_rank * sorted_relevant.to(dtype=torch.float32)).sum(dim=1) / (
                num_positives.to(dtype=torch.float32)
            )
            ap_chunks.append(average_precision)

            for k in k_values:
                effective_k = min(k, max_k)
                hits = sorted_relevant[:, :effective_k].any(dim=1).sum().item()
                recall_hits[k] += int(hits)

    ap_scores = torch.concat(ap_chunks, dim=0)
    mean_average_precision = float(ap_scores.mean().item())
    num_queries = len(query_features)
    recall_at_k = {k: recall_hits[k] / num_queries for k in k_values}

    return (mean_average_precision, recall_at_k)
