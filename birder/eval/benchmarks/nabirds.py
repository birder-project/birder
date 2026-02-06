"""
NABirds benchmark using KNN for bird species classification

Website: https://dl.allaboutbirds.org/nabirds
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.datahub.evaluation import NABirds
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.knn import evaluate_knn

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]], k_values: list[int]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("NABirds (KNN)", style="dim")
    for k in k_values:
        table.add_column(f"k={k}", justify="right")

    for result in results:
        row = [Path(result["embeddings_file"]).name]
        for k in k_values:
            acc = result["accuracies"].get(k)
            row.append(f"{acc:.4f}" if acc is not None else "-")

        table.add_row(*row)

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], k_values: list[int], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "embeddings_file": result["embeddings_file"],
            "method": result["method"],
        }
        for k in k_values:
            row[f"k_{k}_acc"] = result["accuracies"].get(k)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_nabirds_metadata(dataset: NABirds) -> pl.DataFrame:
    images_df = pl.read_csv(dataset.images_path, separator=" ", has_header=False, new_columns=["image_id", "filepath"])
    images_df = images_df.with_columns(
        pl.col("filepath").map_elements(lambda p: Path(p).stem, return_dtype=pl.Utf8).alias("id")
    )
    labels_df = pl.read_csv(dataset.labels_path, separator=" ", has_header=False, new_columns=["image_id", "class_id"])
    classes_df = pl.read_csv(
        dataset.classes_path, separator=" ", has_header=False, new_columns=["class_id", "class_name"]
    )
    split_df = pl.read_csv(
        dataset.train_test_split_path, separator=" ", has_header=False, new_columns=["image_id", "is_train"]
    )

    metadata_df = (
        images_df.join(labels_df, on="image_id")
        .join(classes_df, on="class_id")
        .join(split_df, on="image_id")
        .select(["id", "class_id", "class_name", "is_train"])
    )

    # Create contiguous label indices (0 to num_classes-1)
    unique_classes = metadata_df.get_column("class_id").unique().sort()
    class_id_to_label = {cid: idx for idx, cid in enumerate(unique_classes.to_list())}
    metadata_df = metadata_df.with_columns(pl.col("class_id").replace_strict(class_id_to_label).alias("label"))

    return metadata_df


def _load_embeddings_with_split(
    embeddings_path: str, metadata_df: pl.DataFrame
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_], npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    sample_ids, all_features = load_embeddings(embeddings_path)
    emb_df = pl.DataFrame({"id": sample_ids, "embedding": all_features.tolist()})

    joined = metadata_df.join(emb_df, on="id", how="inner")
    if joined.height < metadata_df.height:
        logger.warning(f"Join dropped {metadata_df.height - joined.height} samples (missing embeddings)")

    all_features = np.array(joined.get_column("embedding").to_list(), dtype=np.float32)
    all_labels = joined.get_column("label").to_numpy().astype(np.int_)
    is_train = joined.get_column("is_train").to_numpy().astype(bool)

    num_classes = all_labels.max() + 1
    logger.info(
        f"Loaded {all_features.shape[0]} samples with {all_features.shape[1]} dimensions, {num_classes} classes"
    )

    x_train = all_features[is_train]
    y_train = all_labels[is_train]
    x_test = all_features[~is_train]
    y_test = all_labels[~is_train]

    logger.info(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")

    return (x_train, y_train, x_test, y_test)


def evaluate_nabirds(args: argparse.Namespace) -> None:
    tic = time.time()

    logger.info(f"Loading NABirds dataset from {args.dataset_path}")
    dataset = NABirds(args.dataset_path)
    metadata_df = _load_nabirds_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_test, y_test = _load_embeddings_with_split(embeddings_path, metadata_df)

        accuracies: dict[int, float] = {}
        for k in args.k:
            logger.info(f"Evaluating KNN with k={k}")
            y_pred, y_true = evaluate_knn(x_train, y_train, x_test, y_test, k=k)
            acc = float(np.mean(y_pred == y_true))
            accuracies[k] = acc
            logger.info(f"k={k} - Accuracy: {acc:.4f}")

        results.append(
            {
                "embeddings_file": str(embeddings_path),
                "method": "knn",
                "accuracies": accuracies,
            }
        )

    _print_summary_table(results, args.k)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("nabirds.csv")
        _write_results_csv(results, args.k, output_path)

    toc = time.time()
    logger.info(f"NABirds benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "nabirds",
        allow_abbrev=False,
        help="run NABirds benchmark - 555 class classification using KNN",
        description="run NABirds benchmark - 555 class classification using KNN",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval nabirds --embeddings "
            "results/vit_b16_224px_crop1.0_48562_embeddings.parquet "
            "--dataset-path ~/Datasets/nabirds --dry-run\n"
            "python -m birder.eval nabirds --embeddings results/nabirds/*.parquet --dataset-path ~/Datasets/nabirds\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to NABirds dataset root")
    subparser.add_argument(
        "--k", type=int, nargs="+", default=[10, 20, 100], help="number of nearest neighbors for KNN"
    )
    subparser.add_argument(
        "--dir", type=str, default="nabirds", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    if args.embeddings is None:
        raise cli.ValidationError("--embeddings is required")
    if args.dataset_path is None:
        raise cli.ValidationError("--dataset-path is required")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_nabirds(args)
