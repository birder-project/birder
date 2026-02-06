"""
AMI clustering for unsupervised embedding evaluation

Paper "An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders",
https://arxiv.org/abs/2406.02465
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score

from birder.eval._embeddings import l2_normalize

logger = logging.getLogger(__name__)

try:
    import umap

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def evaluate_ami(
    features: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int_],
    n_clusters: int,
    umap_dim: int = 50,
    l2_normalize_features: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate embedding quality using UMAP + Agglomerative Clustering + AMI

    Parameters
    ----------
    features
        Feature array of shape (n_samples, embedding_dim).
    labels
        True labels of shape (n_samples,).
    n_clusters
        Number of clusters (should match number of true classes).
    umap_dim
        Target dimensionality for UMAP reduction.
    l2_normalize_features
        If True, applies row-wise L2 normalization before UMAP.
    seed
        Random seed for UMAP reproducibility. When None, uses all available
        cores (n_jobs=-1) but results are non-deterministic. When set, forces n_jobs=1.

    Returns
    -------
    ami_score
        Adjusted Mutual Information score between true labels and cluster assignments.
    """

    assert _HAS_UMAP, "'pip install umap-learn' to use AMI evaluation"

    if seed is None:
        logger.warning("No seed set, UMAP results will be non-deterministic (using n_jobs=-1)")
        n_jobs = -1
    else:
        n_jobs = 1

    if l2_normalize_features is True:
        features = l2_normalize(features)

    reducer = umap.UMAP(n_components=umap_dim, min_dist=0.0, n_jobs=n_jobs, random_state=seed)
    features_reduced = reducer.fit_transform(features)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_assignments = clustering.fit_predict(features_reduced)

    return float(adjusted_mutual_info_score(labels, cluster_assignments))
