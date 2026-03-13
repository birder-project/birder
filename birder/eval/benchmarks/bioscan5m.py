"""
BIOSCAN-5M benchmark using AMI clustering for unsupervised embedding evaluation

Paper "BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity",
https://arxiv.org/abs/2406.12723
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.data.datasets.directory import class_to_idx_from_paths
from birder.data.datasets.directory import make_image_dataset
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.ami import evaluate_ami

logger = logging.getLogger(__name__)


def _results_filename(args: argparse.Namespace) -> str:
    l2_mode = "off" if args.no_l2_normalize is True else "on"
    return f"bioscan5m_ami_umap{args.umap_dim}_l2-{l2_mode}.csv"


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("BIOSCAN-5M (AMI)", style="dim")
    table.add_column("AMI Score", justify="right")
    table.add_column("Classes", justify="right")
    table.add_column("Samples", justify="right")

    for result in sorted(results, key=lambda result: Path(result["embeddings_file"]).name):
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['ami_score']:.4f}",
            str(result["num_classes"]),
            str(result["num_samples"]),
        )

    console.print(table)


def _result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "embeddings_file": result["embeddings_file"],
        "method": result["method"],
        "ami_score": result["ami_score"],
        "l2_normalize": result["l2_normalize"],
        "num_classes": result["num_classes"],
        "num_samples": result["num_samples"],
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
        "ami_score": float(row["ami_score"]),
        "l2_normalize": bool(row["l2_normalize"]),
        "num_classes": int(row["num_classes"]),
        "num_samples": int(row["num_samples"]),
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


def _load_bioscan5m_metadata(data_path: str) -> pl.DataFrame:
    """
    Load metadata from an ImageFolder-compatible directory

    Returns DataFrame with columns: id (filename stem), label
    """

    class_to_idx = class_to_idx_from_paths([data_path])
    image_dataset = make_image_dataset([data_path], class_to_idx)

    rows: list[dict[str, Any]] = []
    for i in range(len(image_dataset)):
        path = image_dataset.paths[i].decode("utf-8")
        label = image_dataset.labels[i].item()
        rows.append({"id": Path(path).stem, "label": label})

    return pl.DataFrame(rows)


def _load_embeddings_with_labels(embeddings_path: str, metadata_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = load_embeddings(embeddings_path)

    joined = metadata_df.join(emb_df, on="id", how="inner")
    if joined.height < metadata_df.height:
        logger.warning(f"Join dropped {metadata_df.height - joined.height} samples (missing embeddings)")

    features = joined.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    labels = joined.get_column("label").to_numpy().astype(np.int_)

    num_classes = len(metadata_df.get_column("label").unique())
    logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} dimensions, {num_classes} classes")

    return (features, labels, num_classes)


def evaluate_bioscan5m(args: argparse.Namespace) -> None:
    tic = time.time()

    logger.info(f"Loading dataset from {args.data_path}")
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
        logger.info(f"BIOSCAN-5M benchmark completed in {lib.format_duration(toc - tic)}")
        return

    metadata_df = _load_bioscan5m_metadata(args.data_path)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

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

        features, labels, num_classes = _load_embeddings_with_labels(embeddings_path, metadata_df)

        logger.info(
            f"Evaluating AMI with umap_dim={args.umap_dim}, seed={args.seed}, "
            f"l2_normalize={not args.no_l2_normalize}"
        )
        ami_score = evaluate_ami(
            features,
            labels,
            n_clusters=num_classes,
            umap_dim=args.umap_dim,
            l2_normalize_features=not args.no_l2_normalize,
            seed=args.seed,
        )
        logger.info(f"AMI score: {ami_score:.4f}")

        results.append(
            {
                "embeddings_file": str(embeddings_path),
                "method": "ami",
                "ami_score": ami_score,
                "l2_normalize": not args.no_l2_normalize,
                "num_classes": num_classes,
                "num_samples": len(labels),
            }
        )
        if output_path is not None:
            result = results[-1]
            _append_result_csv(result, output_path)
            row_df = pl.DataFrame([_result_to_row(result)])
            if existing_df is None:
                existing_df = row_df
            else:
                existing_df = pl.concat([existing_df, row_df], how="diagonal_relaxed")

    _print_summary_table(_summary_results(existing_df, results))

    toc = time.time()
    logger.info(f"BIOSCAN-5M benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "bioscan5m",
        allow_abbrev=False,
        help="run BIOSCAN-5M benchmark - unsupervised embedding evaluation using AMI clustering",
        description="run BIOSCAN-5M benchmark - unsupervised embedding evaluation using AMI clustering",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval bioscan5m --embeddings "
            "results/embeddings.parquet --data-path ~/Datasets/BIOSCAN-5M/species/testing_unseen --dry-run\n"
            "python -m birder.eval bioscan5m --embeddings results/bioscan5m/*.parquet "
            "--data-path ~/Datasets/BIOSCAN-5M/species/testing_unseen --seed 0\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--data-path", type=str, metavar="PATH", help="path to ImageFolder-compatible directory")
    subparser.add_argument("--umap-dim", type=int, default=50, help="target dimensionality for UMAP reduction")
    subparser.add_argument(
        "--no-l2-normalize",
        default=False,
        action="store_true",
        help="disable L2 normalization of embeddings before UMAP",
    )
    subparser.add_argument("--seed", type=int, help="random seed for UMAP")
    subparser.add_argument(
        "--dir", type=str, default="bioscan5m", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    if args.data_path is None:
        raise cli.ValidationError("--data-path is required")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_bioscan5m(args)
