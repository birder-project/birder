"""
ImageNet-1K benchmark using SimpleShot for validation classification

Link: https://image-net.org/challenges/LSVRC/2012/
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
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.simpleshot import evaluate_simpleshot
from birder.eval.methods.simpleshot import sample_k_shot

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]], k_values: list[int]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("ImageNet-1K (SimpleShot)", style="dim")
    for k in k_values:
        table.add_column(f"k={k}", justify="right")

    table.add_column("Runs", justify="right")

    for result in results:
        row = [Path(result["train_embeddings_file"]).name]
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
            "train_embeddings_file": result["train_embeddings_file"],
            "val_embeddings_file": result["val_embeddings_file"],
            "method": result["method"],
            "num_runs": result["num_runs"],
        }
        for k in k_values:
            row[f"k_{k}_acc"] = result["accuracies"].get(k)
            row[f"k_{k}_std"] = result["accuracies_std"].get(k)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_split_parquet(parquet_path: str, split_name: str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    logger.info(f"Loading {split_name} embeddings from {parquet_path}")
    df = load_embeddings(parquet_path, with_label=True)
    features = df.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    labels = df.get_column("label").to_numpy().astype(np.int_, copy=False)

    return (features, labels)


def _evaluate_single_k(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.int_],
    k: int,
    num_runs: int,
    seed: int,
) -> tuple[float, float]:
    logger.info(f"Evaluating k={k} ({k}-shot sampling, SimpleShot)")

    scores: list[float] = []
    for run in range(num_runs):
        run_seed = seed + run
        rng = np.random.default_rng(run_seed)

        x_train_k, y_train_k = sample_k_shot(x_train, y_train, k, rng)
        y_pred, y_true = evaluate_simpleshot(x_train_k, y_train_k, x_val, y_val)

        acc = float(np.mean(y_pred == y_true))
        scores.append(acc)
        logger.info(f"Run {run + 1}/{num_runs} - Accuracy: {acc:.4f}")

    scores_arr = np.array(scores)
    mean_acc = float(scores_arr.mean())
    std_acc = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0
    logger.info(f"k={k} - Mean accuracy over {num_runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")

    return (mean_acc, std_acc)


def evaluate_imagenet1k(args: argparse.Namespace) -> None:
    tic = time.time()

    results: list[dict[str, Any]] = []
    pairs = list(zip(args.train_embeddings, args.val_embeddings, strict=True))
    total = len(pairs)
    for idx, (train_embeddings_path, val_embeddings_path) in enumerate(pairs, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: train={train_embeddings_path}, val={val_embeddings_path}")
        x_train, y_train = _load_split_parquet(train_embeddings_path, split_name="train")
        x_val, y_val = _load_split_parquet(val_embeddings_path, split_name="val")

        accuracies: dict[int, float] = {}
        accuracies_std: dict[int, float] = {}
        for k in args.k:
            mean_acc, std_acc = _evaluate_single_k(x_train, y_train, x_val, y_val, k, args.runs, args.seed)
            accuracies[k] = mean_acc
            accuracies_std[k] = std_acc

        results.append(
            {
                "train_embeddings_file": str(train_embeddings_path),
                "val_embeddings_file": str(val_embeddings_path),
                "method": "simpleshot",
                "num_runs": args.runs,
                "accuracies": accuracies,
                "accuracies_std": accuracies_std,
            }
        )

    _print_summary_table(results, args.k)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("imagenet1k.csv")
        _write_results_csv(results, args.k, output_path)

    toc = time.time()
    logger.info(f"ImageNet-1K benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "imagenet1k",
        allow_abbrev=False,
        help="run ImageNet-1K benchmark - validation classification using SimpleShot",
        description="run ImageNet-1K benchmark - validation classification using SimpleShot",
        epilog=(
            "Generate embeddings:\n"
            "python -m birder.scripts.predict -n hieradet_d_small -t dino-v2 --fast-matmul --batch-size 128 "
            "--chunk-size 50000 --simple-crop --save-embeddings --save-labels --output-format parquet "
            "--prefix imagenet1k --gpu --parallel --wds --wds-size 1281167 /mnt/data/imagenet-1k-wds/training\n"
            "python -m birder.scripts.predict -n hieradet_d_small -t dino-v2 --fast-matmul --batch-size 128 "
            "--simple-crop --save-embeddings --save-labels --output-format parquet --prefix imagenet1k "
            "--gpu --parallel --wds /mnt/data/imagenet-1k-wds/validation\n"
            "\n"
            "Run benchmark:\n"
            "python -m birder.eval imagenet1k --train-embeddings "
            "results/imagenet1k/imagenet1k_model_1281167_sc_embeddings.parquet --val-embeddings "
            "results/imagenet1k/imagenet1k_model_50000_sc_embeddings.parquet --dry-run\n"
            "python -m birder.eval imagenet1k --train-embeddings results/imagenet1k/*1281167*embeddings.parquet "
            "--val-embeddings results/imagenet1k/*50000*embeddings.parquet --k 1 5 --runs 5\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--train-embeddings", type=str, nargs="+", metavar="FILE", help="paths to training embeddings parquet files"
    )
    subparser.add_argument(
        "--val-embeddings", type=str, nargs="+", metavar="FILE", help="paths to validation embeddings parquet files"
    )
    subparser.add_argument(
        "--k", type=int, nargs="+", default=[1, 5, 100], help="number of examples per class for k-shot sampling"
    )
    subparser.add_argument("--runs", type=int, default=5, help="number of evaluation runs")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument(
        "--dir", type=str, default="imagenet1k", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    if args.train_embeddings is None:
        raise cli.ValidationError("--train-embeddings is required")
    if args.val_embeddings is None:
        raise cli.ValidationError("--val-embeddings is required")
    if len(args.train_embeddings) != len(args.val_embeddings):
        raise cli.ValidationError("--train-embeddings and --val-embeddings must contain the same number of files")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_imagenet1k(args)
