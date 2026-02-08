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


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("BIOSCAN-5M (AMI)", style="dim")
    table.add_column("AMI Score", justify="right")
    table.add_column("Classes", justify="right")
    table.add_column("Samples", justify="right")

    for result in results:
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['ami_score']:.4f}",
            str(result["num_classes"]),
            str(result["num_samples"]),
        )

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "embeddings_file": result["embeddings_file"],
                "method": result["method"],
                "ami_score": result["ami_score"],
                "l2_normalize": result["l2_normalize"],
                "num_classes": result["num_classes"],
                "num_samples": result["num_samples"],
            }
        )

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


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
    metadata_df = _load_bioscan5m_metadata(args.data_path)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
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

    _print_summary_table(results)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("bioscan5m.csv")
        _write_results_csv(results, output_path)

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
    if args.embeddings is None:
        raise cli.ValidationError("--embeddings is required")
    if args.data_path is None:
        raise cli.ValidationError("--data-path is required")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_bioscan5m(args)
