"""
Flowers102 benchmark using SimpleShot for flower species classification

Paper "Automated Flower Classification over a Large Number of Classes"
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
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
from torchvision.datasets import ImageFolder

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.simpleshot import evaluate_simpleshot

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("Flowers102 (SimpleShot)", style="dim")
    table.add_column("Val Acc", justify="right")
    table.add_column("Test Acc", justify="right")

    for result in results:
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['val_accuracy']:.4f}",
            f"{result['test_accuracy']:.4f}",
        )

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "embeddings_file": result["embeddings_file"],
                "method": result["method"],
                "val_accuracy": result["val_accuracy"],
                "test_accuracy": result["test_accuracy"],
            }
        )

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_flowers102_metadata(dataset_path: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["training", "validation", "testing"]:
        split_dir = dataset_path.joinpath(split)
        if not split_dir.exists():
            continue

        dataset = ImageFolder(str(split_dir))
        for path, label in dataset.samples:
            rows.append({"id": Path(path).stem, "label": label, "split": split})

    return pl.DataFrame(rows)


def _load_embeddings_with_split(embeddings_path: str, metadata_df: pl.DataFrame) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    sample_ids, all_features = load_embeddings(embeddings_path)
    emb_df = pl.DataFrame({"id": sample_ids, "embedding": all_features.tolist()})

    joined = metadata_df.join(emb_df, on="id", how="inner")
    if joined.height < metadata_df.height:
        logger.warning(f"Join dropped {metadata_df.height - joined.height} samples (missing embeddings)")

    all_features = np.array(joined.get_column("embedding").to_list(), dtype=np.float32)
    all_labels = joined.get_column("label").to_numpy().astype(np.int_)
    splits = joined.get_column("split").to_list()

    is_train = np.array([s == "training" for s in splits], dtype=bool)
    is_val = np.array([s == "validation" for s in splits], dtype=bool)
    is_test = np.array([s == "testing" for s in splits], dtype=bool)

    num_classes = all_labels.max() + 1
    logger.info(
        f"Loaded {all_features.shape[0]} samples with {all_features.shape[1]} dimensions, {num_classes} classes"
    )

    x_train = all_features[is_train]
    y_train = all_labels[is_train]
    x_val = all_features[is_val]
    y_val = all_labels[is_val]
    x_test = all_features[is_test]
    y_test = all_labels[is_test]

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} samples")

    return (x_train, y_train, x_val, y_val, x_test, y_test)


def evaluate_flowers102_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.int_],
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.int_],
    embeddings_path: str,
) -> dict[str, Any]:
    # Evaluate on validation set
    y_pred_val, y_true_val = evaluate_simpleshot(x_train, y_train, x_val, y_val)
    val_acc = float(np.mean(y_pred_val == y_true_val))
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    # Evaluate on test set
    y_pred_test, y_true_test = evaluate_simpleshot(x_train, y_train, x_test, y_test)
    test_acc = float(np.mean(y_pred_test == y_true_test))
    logger.info(f"Test accuracy: {test_acc:.4f}")

    return {
        "method": "simpleshot",
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "embeddings_file": str(embeddings_path),
    }


def evaluate_flowers102(args: argparse.Namespace) -> None:
    tic = time.time()

    logger.info(f"Loading Flowers102 dataset from {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
    metadata_df = _load_flowers102_metadata(dataset_path)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_val, y_val, x_test, y_test = _load_embeddings_with_split(embeddings_path, metadata_df)

        result = evaluate_flowers102_single(x_train, y_train, x_val, y_val, x_test, y_test, embeddings_path)
        results.append(result)

    _print_summary_table(results)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("flowers102.csv")
        _write_results_csv(results, output_path)

    toc = time.time()
    logger.info(f"Flowers102 benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "flowers102",
        allow_abbrev=False,
        help="run Flowers102 benchmark - 102 class classification using SimpleShot",
        description="run Flowers102 benchmark - 102 class classification using SimpleShot",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval flowers102 --embeddings "
            "results/flowers102_embeddings.parquet --dataset-path ~/Datasets/Flowers102 --dry-run\n"
            "python -m birder.eval flowers102 --embeddings results/flowers102/*.parquet "
            "--dataset-path ~/Datasets/Flowers102\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to Flowers102 dataset root")
    subparser.add_argument(
        "--dir", type=str, default="flowers102", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_flowers102(args)
