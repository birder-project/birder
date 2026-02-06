"""
FishNet benchmark using MLP probe for multi-label trait prediction

Paper "FishNet: A Large-scale Dataset and Benchmark for Fish Recognition, Detection, and Functional Trait Prediction"
https://fishnet-2023.github.io/
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
from sklearn.metrics import f1_score

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.datahub.evaluation import FishNet
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.mlp import evaluate_mlp
from birder.eval.methods.mlp import train_mlp

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("FishNet (MLP)", style="dim")
    table.add_column("Macro F1", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Exact Match", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Runs", justify="right")
    for result in results:
        row = [
            Path(result["embeddings_file"]).name,
            f"{result['macro_f1']:.4f}",
            f"{result['macro_f1_std']:.4f}",
            f"{result['exact_match_acc']:.4f}",
            f"{result['exact_match_acc_std']:.4f}",
            f"{result['num_runs']}",
        ]

        table.add_row(*row)

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], trait_names: list[str], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "embeddings_file": result["embeddings_file"],
            "method": result["method"],
            "macro_f1": result["macro_f1"],
            "macro_f1_std": result["macro_f1_std"],
            "exact_match_acc": result["exact_match_acc"],
            "exact_match_acc_std": result["exact_match_acc_std"],
            "num_runs": result["num_runs"],
        }
        for trait in trait_names:
            row[f"f1_{trait}"] = result["per_trait_f1"].get(trait)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_fishnet_data(csv_path: Path, trait_columns: list[str]) -> pl.DataFrame:
    """
    Load FishNet CSV and prepare metadata

    Returns DataFrame with columns: id, trait labels (0/1)
    """

    df = pl.read_csv(csv_path)
    df = df.with_columns(
        pl.col("image")
        .str.extract(r"([^/]+)$")  # Get filename (last path segment)
        .str.replace(r"\.[^.]+$", "")  # Remove extension
        .alias("id")
    )

    # Encode FeedingPath: benthic=0, pelagic=1
    df = df.with_columns(
        pl.when(pl.col("FeedingPath") == "pelagic")
        .then(pl.lit(1))
        .when(pl.col("FeedingPath") == "benthic")
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .alias("FeedingPath_encoded")
    )

    # Select relevant columns
    other_traits = [t for t in trait_columns if t != "FeedingPath"]
    select_cols = ["id", "FeedingPath_encoded"] + other_traits
    df = df.select(select_cols)

    # Rename FeedingPath_encoded back to FeedingPath
    df = df.rename({"FeedingPath_encoded": "FeedingPath"})

    # Filter rows with any null trait values
    for trait in trait_columns:
        df = df.filter(pl.col(trait).is_not_null())

    return df


def _load_embeddings_with_labels(
    embeddings_path: str, train_df: pl.DataFrame, test_df: pl.DataFrame, trait_columns: list[str]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    sample_ids, all_features = load_embeddings(embeddings_path)
    emb_df = pl.DataFrame({"id": sample_ids, "embedding": all_features.tolist()})

    # Join with train data
    train_joined = train_df.join(emb_df, on="id", how="inner")
    if train_joined.height < train_df.height:
        logger.warning(f"Train: dropped {train_df.height - train_joined.height} samples (missing embeddings)")

    # Join with test data
    test_joined = test_df.join(emb_df, on="id", how="inner")
    if test_joined.height < test_df.height:
        logger.warning(f"Test: dropped {test_df.height - test_joined.height} samples (missing embeddings)")

    # Extract features and labels
    x_train = np.array(train_joined.get_column("embedding").to_list(), dtype=np.float32)
    y_train = train_joined.select(trait_columns).to_numpy().astype(np.float32)

    x_test = np.array(test_joined.get_column("embedding").to_list(), dtype=np.float32)
    y_test = test_joined.select(trait_columns).to_numpy().astype(np.float32)

    logger.info(f"Train: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")
    logger.info(f"Features: {x_train.shape[1]} dims, Traits: {len(trait_columns)}")

    return (x_train, y_train, x_test, y_test)


# pylint: disable=too-many-locals
def evaluate_fishnet_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.float32],
    trait_columns: list[str],
    args: argparse.Namespace,
    embeddings_path: str,
    device: torch.device,
) -> dict[str, Any]:
    num_classes = len(trait_columns)

    scores: list[float] = []
    exact_match_scores: list[float] = []
    per_trait_f1_runs: list[dict[str, float]] = []

    for run in range(args.runs):
        run_seed = args.seed + run
        logger.info(f"Run {run + 1}/{args.runs} (seed={run_seed})")

        # Train MLP
        model = train_mlp(
            x_train,
            y_train,
            num_classes=num_classes,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=run_seed,
        )

        # Evaluate
        y_pred, y_true, macro_f1 = evaluate_mlp(model, x_test, y_test, batch_size=args.batch_size, device=device)
        scores.append(macro_f1)
        exact_match_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))
        exact_match_scores.append(exact_match_acc)

        # Per-trait F1
        per_trait_f1: dict[str, float] = {}
        for i, trait in enumerate(trait_columns):
            trait_f1 = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0.0)
            per_trait_f1[trait] = float(trait_f1)

        per_trait_f1_runs.append(per_trait_f1)
        logger.info(f"Run {run + 1}/{args.runs} - Macro F1: {macro_f1:.4f}, Exact Match: {exact_match_acc:.4f}")

    # Average results
    scores_arr = np.array(scores)
    mean_f1 = float(scores_arr.mean())
    std_f1 = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0
    exact_scores_arr = np.array(exact_match_scores)
    mean_exact = float(exact_scores_arr.mean())
    std_exact = float(exact_scores_arr.std(ddof=1)) if len(exact_match_scores) > 1 else 0.0

    # Average per-trait F1 across runs
    avg_per_trait_f1: dict[str, float] = {}
    for trait in trait_columns:
        trait_scores = [run_f1[trait] for run_f1 in per_trait_f1_runs]
        avg_per_trait_f1[trait] = float(np.mean(trait_scores))

    logger.info(f"Mean Macro F1 over {args.runs} runs: {mean_f1:.4f} +/- {std_f1:.4f} (std)")
    logger.info(f"Mean Exact Match over {args.runs} runs: {mean_exact:.4f} +/- {std_exact:.4f} (std)")
    for trait, f1 in avg_per_trait_f1.items():
        logger.info(f"  {trait}: {f1:.4f}")

    return {
        "method": "mlp",
        "macro_f1": mean_f1,
        "macro_f1_std": std_f1,
        "exact_match_acc": mean_exact,
        "exact_match_acc_std": std_exact,
        "num_runs": args.runs,
        "per_trait_f1": avg_per_trait_f1,
        "embeddings_file": str(embeddings_path),
    }


def evaluate_fishnet(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    logger.info(f"Loading FishNet dataset from {args.dataset_path}")
    dataset = FishNet(args.dataset_path)
    trait_columns = dataset.trait_columns

    train_df = _load_fishnet_data(dataset.train_csv, trait_columns)
    test_df = _load_fishnet_data(dataset.test_csv, trait_columns)
    logger.info(f"Train samples: {train_df.height}, Test samples: {test_df.height}")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_test, y_test = _load_embeddings_with_labels(
            embeddings_path, train_df, test_df, trait_columns
        )

        result = evaluate_fishnet_single(x_train, y_train, x_test, y_test, trait_columns, args, embeddings_path, device)
        results.append(result)

    _print_summary_table(results)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("fishnet.csv")
        _write_results_csv(results, trait_columns, output_path)

    toc = time.time()
    logger.info(f"FishNet benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "fishnet",
        allow_abbrev=False,
        help="run FishNet benchmark - 9 trait multi-label classification using MLP probe",
        description="run FishNet benchmark - 9 trait multi-label classification using MLP probe",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval fishnet --embeddings "
            "results/vit_b16_224px_embeddings.parquet "
            "--dataset-path ~/Datasets/fishnet --dry-run\n"
            "python -m birder.eval fishnet --embeddings results/fishnet/*.parquet "
            "--dataset-path ~/Datasets/fishnet --gpu --gpu-id 1\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to FishNet dataset root")
    subparser.add_argument("--runs", type=int, default=3, help="number of evaluation runs")
    subparser.add_argument("--epochs", type=int, default=100, help="training epochs per run")
    subparser.add_argument("--batch-size", type=int, default=128, help="batch size for training and inference")
    subparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    subparser.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden layer dimension")
    subparser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="fishnet", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_fishnet(args)
