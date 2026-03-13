"""
Butterflies benchmark using KNN for butterfly and moth species classification

Dataset "Butterflies and Moths Austria",
https://huggingface.co/datasets/birder-project/butterflies-moths-austria
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.eval.methods.knn import evaluate_knn

logger = logging.getLogger(__name__)

_EVAL_SPLITS = ("training", "validation", "testing")


def _results_filename(args: argparse.Namespace) -> str:
    return f"butterflies_knn_k{args.k}.csv"


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("Butterflies and Moths (KNN)", style="dim")
    table.add_column("Val Macro F1", justify="right")
    table.add_column("Test Macro F1", justify="right")
    table.add_column("Classes", justify="right")

    for result in sorted(results, key=lambda result: Path(result["embeddings_file"]).name):
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['val_macro_f1']:.4f}",
            f"{result['test_macro_f1']:.4f}",
            str(result["num_classes"]),
        )

    console.print(table)


def _result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "embeddings_file": result["embeddings_file"],
        "method": result["method"],
        "k": result["k"],
        "val_macro_f1": result["val_macro_f1"],
        "test_macro_f1": result["test_macro_f1"],
        "num_classes": result["num_classes"],
        "train_samples": result["train_samples"],
        "val_samples": result["val_samples"],
        "test_samples": result["test_samples"],
    }


def _append_result_csv(result: dict[str, Any], output_path: Path) -> None:
    row_df = pl.DataFrame([_result_to_row(result)])
    file_exists = output_path.exists()
    mode = "a" if file_exists is True else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        row_df.write_csv(handle, include_header=file_exists is False)

    logger.info(f"Results updated at {output_path}")


def _row_to_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "embeddings_file": str(row["embeddings_file"]),
        "method": str(row["method"]),
        "k": int(row["k"]),
        "val_macro_f1": float(row["val_macro_f1"]),
        "test_macro_f1": float(row["test_macro_f1"]),
        "num_classes": int(row["num_classes"]),
        "train_samples": int(row["train_samples"]),
        "val_samples": int(row["val_samples"]),
        "test_samples": int(row["test_samples"]),
    }


def _load_cached_result(existing_df: Optional[pl.DataFrame], embeddings_path: str) -> Optional[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return None

    match_df = existing_df.filter(pl.col("embeddings_file") == embeddings_path)
    if match_df.is_empty() is True:
        return None

    return _row_to_result(match_df.row(0, named=True))


def _summary_results(
    existing_df: Optional[pl.DataFrame], current_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return current_results

    return [_row_to_result(row) for row in existing_df.iter_rows(named=True)]


def _load_embeddings_with_split(embeddings_path: str) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
    dict[str, int],
]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = pl.read_parquet(embeddings_path).select(
        pl.col("sample").str.replace_all(r"\\", "/").alias("sample"),
        pl.col("embedding"),
    )
    emb_df = emb_df.with_columns(
        pl.col("sample").str.extract(r"/([^/]+)/[^/]+/[^/]+$", 1).alias("split"),
        pl.col("sample").str.extract(r"/[^/]+/([^/]+)/[^/]+$", 1).alias("class_name"),
    )

    parse_failures = emb_df.filter(pl.col("split").is_null() | pl.col("class_name").is_null()).height
    if parse_failures > 0:
        logger.warning(f"Dropping {parse_failures} samples with unparseable split or class from sample path")
        emb_df = emb_df.filter(pl.col("split").is_not_null() & pl.col("class_name").is_not_null())

    ignored_df = emb_df.filter(~pl.col("split").is_in(_EVAL_SPLITS))
    if ignored_df.height > 0:
        ignored_counts = ", ".join(
            f"{row['split']}={row['len']}"
            for row in ignored_df.group_by("split").len().sort("split").iter_rows(named=True)
        )
        logger.info(f"Dropping {ignored_df.height} samples from non-eval splits: {ignored_counts}")

    emb_df = emb_df.filter(pl.col("split").is_in(_EVAL_SPLITS))
    if emb_df.is_empty() is True:
        raise RuntimeError("No evaluation samples found after filtering to training/validation/testing splits")

    class_names = emb_df.get_column("class_name").unique().sort().to_list()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    emb_df = emb_df.with_columns(pl.col("class_name").replace_strict(class_to_idx).alias("label"))

    train_df = emb_df.filter(pl.col("split") == "training")
    val_df = emb_df.filter(pl.col("split") == "validation")
    test_df = emb_df.filter(pl.col("split") == "testing")
    if train_df.is_empty() is True or val_df.is_empty() is True or test_df.is_empty() is True:
        raise RuntimeError("Expected non-empty training, validation, and testing splits")

    x_train = train_df.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_train = train_df.get_column("label").to_numpy().astype(np.int_, copy=False)
    x_val = val_df.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_val = val_df.get_column("label").to_numpy().astype(np.int_, copy=False)
    x_test = test_df.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_test = test_df.get_column("label").to_numpy().astype(np.int_, copy=False)

    stats = {
        "num_classes": len(class_names),
        "train_samples": int(train_df.height),
        "val_samples": int(val_df.height),
        "test_samples": int(test_df.height),
    }
    total_samples = stats["train_samples"] + stats["val_samples"] + stats["test_samples"]
    embedding_dim = x_train.shape[1]
    logger.info(f"Loaded {total_samples} samples with {embedding_dim} dimensions, {stats['num_classes']} classes")
    logger.info(
        f"Train: {stats['train_samples']} samples, Val: {stats['val_samples']} samples, "
        f"Test: {stats['test_samples']} samples"
    )

    return (x_train, y_train, x_val, y_val, x_test, y_test, stats)


def evaluate_butterflies_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.int_],
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.int_],
    embeddings_path: str,
    stats: dict[str, int],
    k: int,
    device: torch.device,
) -> dict[str, Any]:
    logger.info(f"Evaluating KNN with k={k}")

    y_pred_val, y_true_val = evaluate_knn(x_train, y_train, x_val, y_val, k=k, device=device)
    val_macro_f1 = float(f1_score(y_true_val, y_pred_val, average="macro", zero_division=0.0))
    logger.info(f"Validation macro F1: {val_macro_f1:.4f}")

    y_pred_test, y_true_test = evaluate_knn(x_train, y_train, x_test, y_test, k=k, device=device)
    test_macro_f1 = float(f1_score(y_true_test, y_pred_test, average="macro", zero_division=0.0))
    logger.info(f"Test macro F1: {test_macro_f1:.4f}")

    return {
        "embeddings_file": str(embeddings_path),
        "method": "knn",
        "k": k,
        "val_macro_f1": val_macro_f1,
        "test_macro_f1": test_macro_f1,
        **stats,
    }


def evaluate_butterflies(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    existing_df: Optional[pl.DataFrame] = None
    output_path: Optional[Path] = None

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath(_results_filename(args))
        if output_path.exists() is True:
            logger.info(f"Loading existing results from {output_path}")
            existing_df = pl.read_csv(output_path)

    if args.embeddings is None:
        summary_results = _summary_results(existing_df, [])
        if len(summary_results) == 0:
            raise cli.ValidationError("--embeddings is required when no cached results are available")

        logger.info("No embeddings provided, showing cached results")
        _print_summary_table(summary_results)
        toc = time.time()
        logger.info(f"Butterflies benchmark completed in {lib.format_duration(toc - tic)}")
        return

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")

        cached_result = _load_cached_result(existing_df, embeddings_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {embeddings_path}")
            results.append(cached_result)
            continue

        if Path(embeddings_path).exists() is False:
            logger.warning(f"Embeddings file not found, skipping: {embeddings_path}")
            continue

        x_train, y_train, x_val, y_val, x_test, y_test, stats = _load_embeddings_with_split(embeddings_path)
        result = evaluate_butterflies_single(
            x_train, y_train, x_val, y_val, x_test, y_test, embeddings_path, stats, args.k, device
        )
        results.append(result)
        if output_path is not None:
            _append_result_csv(result, output_path)
            row_df = pl.DataFrame([_result_to_row(result)])
            if existing_df is None:
                existing_df = row_df
            else:
                existing_df = pl.concat([existing_df, row_df], how="diagonal_relaxed")

    _print_summary_table(_summary_results(existing_df, results))

    toc = time.time()
    logger.info(f"Butterflies benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "butterflies",
        allow_abbrev=False,
        help="run Butterflies benchmark - species classification using KNN",
        description="run Butterflies benchmark - species classification using KNN",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval butterflies --embeddings "
            "results/butterflies/*.parquet --dry-run\n"
            "python -m birder.eval butterflies --embeddings "
            "results/butterflies/*.parquet --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--k", type=int, default=10, help="number of neighbors for KNN evaluation")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="butterflies", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    evaluate_butterflies(args)
