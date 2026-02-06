import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

logger = logging.getLogger(__name__)


def l2_normalize(x: npt.NDArray[np.float32], eps: float = 1e-12) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)  # type: ignore[no-any-return]


def load_embeddings(path: Path | str) -> tuple[list[str], npt.NDArray[np.float32]]:
    """
    Load embeddings from parquet file

    Auto-detects format:
    - If 'embedding' column exists: use directly
    - If numeric column names (0, 1, 2, ...): treat as logits, convert to array

    Returns
    -------
    sample_ids
        List of sample identifiers (stem of 'sample' column path).
    features
        Array of shape (n_samples, embedding_dim), dtype float32.
    """

    if isinstance(path, str):
        path = Path(path)

    df = pl.read_parquet(path)
    df = df.with_columns(pl.col("sample").map_elements(lambda p: Path(p).stem, return_dtype=pl.Utf8).alias("id"))

    if "embedding" in df.columns:
        df = df.select(["id", "embedding"])
    else:
        # Logits format - numeric column names
        embed_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
        df = df.with_columns(
            pl.concat_list(pl.col(embed_cols)).cast(pl.Array(pl.Float32, len(embed_cols))).alias("embedding")
        ).select(["id", "embedding"])

    sample_ids = df.get_column("id").to_list()
    features = df.get_column("embedding").to_numpy().astype(np.float32, copy=False)

    return (sample_ids, features)
