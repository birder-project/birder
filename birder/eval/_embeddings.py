import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

logger = logging.getLogger(__name__)


def l2_normalize(x: npt.NDArray[np.float32], eps: float = 1e-12) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)

    return x / norms  # type: ignore[no-any-return]


def l2_normalize_(x: npt.NDArray[np.float32], eps: float = 1e-12) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms

    return x


def load_embeddings(path: Path | str) -> pl.DataFrame:
    """
    Load embeddings from parquet file as a dataframe

    Auto-detects format:
    - If 'embedding' column exists: use directly
    - If numeric column names (0, 1, 2, ...): treat as logits, convert to array

    Parameters
    ----------
    path
        Path to parquet embedding or logits file.

    Returns
    -------
    DataFrame with columns:
    - id: sample identifier (stem of 'sample' column path)
    - embedding: fixed-size float32 embedding array
    """

    if isinstance(path, str):
        path = Path(path)

    df = pl.read_parquet(path)
    id_expr = pl.col("sample").str.split("/").list.last().str.replace(r"\.[^./\\]+$", "").alias("id")
    if "embedding" in df.columns:
        return df.select(id_expr, pl.col("embedding"))

    # Logits format - numeric column names
    embed_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    return df.select(
        id_expr,
        pl.concat_arr(pl.col(embed_cols)).cast(pl.Array(pl.Float32, len(embed_cols))).alias("embedding"),
    )
