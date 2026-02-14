"""
FungiCLEF2023 benchmark using KNN for fungi species classification

Link: https://www.imageclef.org/FungiCLEF2023
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.datahub.evaluation import FungiCLEF2023
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.knn import evaluate_knn
from birder.eval.methods.simpleshot import sample_k_shot

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]], k_values: list[int]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("FungiCLEF2023 (KNN)", style="dim")
    for k in k_values:
        table.add_column(f"k={k}", justify="right")

    table.add_column("Runs", justify="right")

    for result in results:
        row = [Path(result["embeddings_file"]).name]
        for k in k_values:
            acc = result["accuracies"].get(k)
            row.append(f"{acc:.4f}" if acc is not None else "-")

        row.append(f"{result['num_runs']}")
        table.add_row(*row)

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], k_values: list[int], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "embeddings_file": result["embeddings_file"],
            "method": result["method"],
            "num_runs": result["num_runs"],
        }
        for k in k_values:
            row[f"k_{k}_acc"] = result["accuracies"].get(k)
            row[f"k_{k}_std"] = result["accuracies_std"].get(k)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_fungiclef_metadata(dataset: FungiCLEF2023) -> pl.DataFrame:
    """
    Load metadata from FungiCLEF2023 CSV files

    Returns DataFrame with columns: id (filename stem), label, split (train/val/test).
    Filters out validation samples with unknown species (class_id == settings.NO_LABEL).
    Test samples have label=settings.NO_LABEL (no ground truth available) and are excluded from evaluation.
    """

    train_df = pl.read_csv(dataset.train_metadata_path)
    train_df = train_df.with_columns(
        pl.col("image_path").map_elements(lambda p: Path(p).stem, return_dtype=pl.Utf8).alias("id"),
        pl.lit("train").alias("split"),
    ).select(["id", "class_id", "split"])

    val_df = pl.read_csv(dataset.val_metadata_path)
    val_df = val_df.filter(pl.col("class_id") != settings.NO_LABEL)
    val_df = val_df.with_columns(
        pl.col("filename").alias("id"),
        pl.lit("val").alias("split"),
    ).select(["id", "class_id", "split"])

    # Include test IDs so they are properly excluded when embeddings contain all samples
    test_df = pl.read_csv(dataset.test_metadata_path)
    test_df = test_df.with_columns(
        pl.col("filename").alias("id"),
        pl.lit("test").alias("split"),
        pl.lit(settings.NO_LABEL, dtype=pl.Int64).alias("class_id"),
    ).select(["id", "class_id", "split"])

    metadata_df = pl.concat([train_df, val_df, test_df])
    metadata_df = metadata_df.rename({"class_id": "label"})

    return metadata_df


def _load_embeddings_with_split(
    embeddings_path: str, metadata_df: pl.DataFrame
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_], npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = load_embeddings(embeddings_path)

    train_meta = metadata_df.filter(pl.col("split") == "train").select(["id", "label"])
    val_meta = metadata_df.filter(pl.col("split") == "val").select(["id", "label"])
    test_meta = metadata_df.filter(pl.col("split") == "test").select(["id"])

    train_join = train_meta.join(emb_df, on="id", how="inner")
    val_join = val_meta.join(emb_df, on="id", how="inner")
    test_join = test_meta.join(emb_df, on="id", how="inner")

    dropped_train = train_meta.height - train_join.height
    dropped_val = val_meta.height - val_join.height
    dropped_test = test_meta.height - test_join.height
    dropped_total = dropped_train + dropped_val + dropped_test
    if dropped_total > 0:
        logger.warning(
            f"Join dropped {dropped_total} samples (missing embeddings): "
            f"train={dropped_train}, val={dropped_val}, test={dropped_test}"
        )

    x_train = train_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_train = train_join.get_column("label").to_numpy().astype(np.int_, copy=False)
    x_val = val_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_val = val_join.get_column("label").to_numpy().astype(np.int_, copy=False)

    num_classes = y_train.max() + 1
    total_samples = x_train.shape[0] + x_val.shape[0] + test_join.height
    embedding_dim = x_train.shape[1]
    logger.info(f"Loaded {total_samples} samples with {embedding_dim} dimensions, {num_classes} classes")

    logger.info(
        f"Train: {len(y_train)} samples, Val: {len(y_val)} samples, "
        f"Test: {test_join.height} samples (no labels, excluded)"
    )

    return (x_train, y_train, x_val, y_val)


def _evaluate_single_k(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.int_],
    k: int,
    num_runs: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    logger.info(f"Evaluating k={k} ({k}-shot sampling, KNN k={k})")

    scores: list[float] = []
    for run in range(num_runs):
        run_seed = seed + run
        rng = np.random.default_rng(run_seed)

        # Sample k examples per class
        x_train_k, y_train_k = sample_k_shot(x_train, y_train, k, rng)

        # Evaluate using KNN with k neighbors
        y_pred, y_true = evaluate_knn(x_train_k, y_train_k, x_val, y_val, k=k, device=device)

        acc = float(np.mean(y_pred == y_true))
        scores.append(acc)
        logger.info(f"Run {run + 1}/{num_runs} - Accuracy: {acc:.4f}")

    scores_arr = np.array(scores)
    mean_acc = float(scores_arr.mean())
    std_acc = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0

    logger.info(f"k={k} - Mean accuracy over {num_runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")

    return (mean_acc, std_acc)


def evaluate_fungiclef(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    logger.info(f"Loading FungiCLEF2023 dataset from {args.dataset_path}")
    dataset = FungiCLEF2023(args.dataset_path)
    metadata_df = _load_fungiclef_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_val, y_val = _load_embeddings_with_split(embeddings_path, metadata_df)

        accuracies: dict[int, float] = {}
        accuracies_std: dict[int, float] = {}
        for k in args.k:
            mean_acc, std_acc = _evaluate_single_k(x_train, y_train, x_val, y_val, k, args.runs, args.seed, device)
            accuracies[k] = mean_acc
            accuracies_std[k] = std_acc

        results.append(
            {
                "embeddings_file": str(embeddings_path),
                "method": "knn",
                "num_runs": args.runs,
                "accuracies": accuracies,
                "accuracies_std": accuracies_std,
            }
        )

    _print_summary_table(results, args.k)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("fungiclef.csv")
        _write_results_csv(results, args.k, output_path)

    toc = time.time()
    logger.info(f"FungiCLEF2023 benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "fungiclef",
        allow_abbrev=False,
        help="run FungiCLEF2023 benchmark - 1,604 species classification using KNN",
        description="run FungiCLEF2023 benchmark - 1,604 species classification using KNN",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval fungiclef --embeddings "
            "results/fungiclef_embeddings.parquet --dataset-path ~/Datasets/FungiCLEF2023 --dry-run\n"
            "python -m birder.eval fungiclef --embeddings results/fungiclef/*.parquet "
            "--dataset-path ~/Datasets/FungiCLEF2023 --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to FungiCLEF2023 dataset root")
    subparser.add_argument(
        "--k", type=int, nargs="+", default=[1, 3], help="k value for k-shot sampling and KNN neighbors"
    )
    subparser.add_argument("--runs", type=int, default=5, help="number of evaluation runs")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="fungiclef", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_fungiclef(args)
