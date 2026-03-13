"""
CUB-200-2011 benchmark using embedding retrieval with mAP and Recall@K

Website: https://www.vision.caltech.edu/datasets/cub_200_2011/
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
from torchvision.datasets import ImageFolder

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.retrieval import evaluate_retrieval

logger = logging.getLogger(__name__)


def _results_filename(args: argparse.Namespace) -> str:
    k_values = "-".join(str(k) for k in args.k)
    return f"cub200_retrieval_k{k_values}.csv"


def _print_summary_table(results: list[dict[str, Any]], k_values: list[int]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("CUB-200-2011 (Retrieval)", style="dim")
    table.add_column("mAP", justify="right")
    for k in k_values:
        table.add_column(f"R@{k}", justify="right")

    for result in sorted(results, key=lambda result: Path(result["embeddings_file"]).name):
        row = [
            Path(result["embeddings_file"]).name,
            f"{result['mean_average_precision']:.4f}",
        ]
        for k in k_values:
            recall = result["recall_at_k"].get(k)
            row.append(f"{recall:.4f}" if recall is not None else "-")

        table.add_row(*row)

    console.print(table)


def _result_to_row(result: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "embeddings_file": result["embeddings_file"],
        "method": result["method"],
        "mean_average_precision": result["mean_average_precision"],
        "num_classes": result["num_classes"],
        "gallery_samples": result["gallery_samples"],
        "query_samples": result["query_samples"],
    }
    for k in k_values:
        row[f"r_at_{k}"] = result["recall_at_k"].get(k)

    return row


def _append_result_csv(result: dict[str, Any], k_values: list[int], output_path: Path) -> None:
    row_df = pl.DataFrame([_result_to_row(result, k_values)])
    file_exists = output_path.exists()
    mode = "a" if file_exists is True else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        row_df.write_csv(handle, include_header=file_exists is False)

    logger.info(f"Results updated at {output_path}")


def _row_to_result(row: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    recall_at_k = {k: float(value) for k in k_values if (value := row.get(f"r_at_{k}")) is not None}

    return {
        "embeddings_file": str(row["embeddings_file"]),
        "method": str(row["method"]),
        "mean_average_precision": float(row["mean_average_precision"]),
        "num_classes": int(row["num_classes"]),
        "gallery_samples": int(row["gallery_samples"]),
        "query_samples": int(row["query_samples"]),
        "recall_at_k": recall_at_k,
    }


def _load_cached_result(
    existing_df: Optional[pl.DataFrame], embeddings_path: str, k_values: list[int]
) -> Optional[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return None

    match_df = existing_df.filter(pl.col("embeddings_file") == embeddings_path)
    if match_df.is_empty() is True:
        return None

    return _row_to_result(match_df.row(0, named=True), k_values)


def _summary_results(
    existing_df: Optional[pl.DataFrame], current_results: list[dict[str, Any]], k_values: list[int]
) -> list[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return current_results

    return [_row_to_result(row, k_values) for row in existing_df.iter_rows(named=True)]


def _load_cub200_metadata(dataset_path: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["training", "validation"]:
        split_dir = dataset_path.joinpath(split)
        if split_dir.exists() is False:
            continue

        dataset = ImageFolder(str(split_dir))
        for path, label in dataset.samples:
            rows.append({"id": Path(path).stem, "label": label, "split": split})

    return pl.DataFrame(rows)


def _load_embeddings_with_split(
    embeddings_path: str, metadata_df: pl.DataFrame
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int_], npt.NDArray[np.float32], npt.NDArray[np.int_], dict[str, int]
]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = load_embeddings(embeddings_path)

    gallery_meta = metadata_df.filter(pl.col("split") == "training").select(["id", "label"])
    query_meta = metadata_df.filter(pl.col("split") == "validation").select(["id", "label"])

    gallery_join = gallery_meta.join(emb_df, on="id", how="inner")
    query_join = query_meta.join(emb_df, on="id", how="inner")

    dropped_gallery = gallery_meta.height - gallery_join.height
    dropped_query = query_meta.height - query_join.height
    dropped_total = dropped_gallery + dropped_query
    if dropped_total > 0:
        logger.warning(
            f"Join dropped {dropped_total} samples (missing embeddings): "
            f"gallery={dropped_gallery}, query={dropped_query}"
        )

    x_gallery = gallery_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_gallery = gallery_join.get_column("label").to_numpy().astype(np.int_)
    x_query = query_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_query = query_join.get_column("label").to_numpy().astype(np.int_)

    stats = {
        "num_classes": int(metadata_df.get_column("label").max() + 1),  # type: ignore[operator]
        "gallery_samples": int(x_gallery.shape[0]),
        "query_samples": int(x_query.shape[0]),
    }
    total_samples = stats["gallery_samples"] + stats["query_samples"]
    embedding_dim = x_gallery.shape[1]
    logger.info(f"Loaded {total_samples} samples with {embedding_dim} dimensions, {stats['num_classes']} classes")
    logger.info(f"Gallery: {stats['gallery_samples']} samples, Query: {stats['query_samples']} samples")

    return (x_gallery, y_gallery, x_query, y_query, stats)


def evaluate_cub200_single(
    x_gallery: npt.NDArray[np.float32],
    y_gallery: npt.NDArray[np.int_],
    x_query: npt.NDArray[np.float32],
    y_query: npt.NDArray[np.int_],
    embeddings_path: str,
    k_values: list[int],
    device: torch.device,
    stats: dict[str, int],
) -> dict[str, Any]:
    logger.info(f"Evaluating retrieval with k={k_values}")
    mean_average_precision, recall_at_k = evaluate_retrieval(
        x_gallery, y_gallery, x_query, y_query, k_values, device=device
    )
    logger.info(f"mAP: {mean_average_precision:.4f}")
    for k in k_values:
        logger.info(f"Recall@{k}: {recall_at_k[k]:.4f}")

    return {
        "embeddings_file": str(embeddings_path),
        "method": "retrieval",
        "mean_average_precision": mean_average_precision,
        "recall_at_k": recall_at_k,
        **stats,
    }


def evaluate_cub200(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    logger.info(f"Loading CUB-200-2011 dataset from {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
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
        summary_results = _summary_results(existing_df, [], args.k)
        if len(summary_results) == 0:
            raise cli.ValidationError("--embeddings is required when no cached results are available")

        logger.info("No embeddings provided, showing cached results")
        _print_summary_table(summary_results, args.k)
        toc = time.time()
        logger.info(f"CUB-200-2011 benchmark completed in {lib.format_duration(toc - tic)}")
        return

    metadata_df = _load_cub200_metadata(dataset_path)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")

        cached_result = _load_cached_result(existing_df, embeddings_path, args.k)
        if cached_result is not None:
            logger.info(f"Using cached result for {embeddings_path}")
            results.append(cached_result)
            continue

        if Path(embeddings_path).exists() is False:
            logger.warning(f"Embeddings file not found, skipping: {embeddings_path}")
            continue

        x_gallery, y_gallery, x_query, y_query, stats = _load_embeddings_with_split(embeddings_path, metadata_df)
        result = evaluate_cub200_single(
            x_gallery,
            y_gallery,
            x_query,
            y_query,
            embeddings_path,
            args.k,
            device,
            stats,
        )
        results.append(result)
        if output_path is not None:
            _append_result_csv(result, args.k, output_path)
            row_df = pl.DataFrame([_result_to_row(result, args.k)])
            if existing_df is None:
                existing_df = row_df
            else:
                existing_df = pl.concat([existing_df, row_df], how="diagonal_relaxed")

    _print_summary_table(_summary_results(existing_df, results, args.k), args.k)

    toc = time.time()
    logger.info(f"CUB-200-2011 benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "cub200",
        allow_abbrev=False,
        help="run CUB-200-2011 benchmark - image retrieval with mAP and Recall@K",
        description="run CUB-200-2011 benchmark - image retrieval with mAP and Recall@K",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval cub200 --embeddings results/cub200_embeddings.parquet "
            "--dataset-path ~/Datasets/CUB_200_2011 --dry-run\n"
            "python -m birder.eval cub200 --embeddings results/cub200/*.parquet "
            "--dataset-path ~/Datasets/CUB_200_2011 --k 1 5 10 --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to CUB-200-2011 dataset root")
    subparser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="Recall@K values to report")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="cub200", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset_path is None:
        raise cli.ValidationError("--dataset-path is required")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_cub200(args)
