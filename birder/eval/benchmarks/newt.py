"""
NeWT benchmark, adapted from
https://github.com/samuelstevens/biobench/blob/main/src/biobench/newt/__init__.py

Paper "Benchmarking Representation Learning for Natural World Image Collections",
https://arxiv.org/abs/2103.16483
"""

# Reference license: MIT

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
from birder.datahub.evaluation import NeWT
from birder.eval._embeddings import l2_normalize
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.simpleshot import normalize_features
from birder.eval.methods.svm import evaluate_svm

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    cluster_names = sorted({cluster for result in results for cluster in result["per_cluster_accuracy"].keys()})

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("NeWT (SVM)", style="dim")
    table.add_column("Accuracy", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Runs", justify="right")
    for cluster in cluster_names:
        table.add_column(cluster.replace("_", " ").title(), justify="right")

    for result in results:
        row = [
            Path(result["embeddings_file"]).name,
            f"{result['accuracy']:.4f}",
            f"{result['accuracy_std']:.4f}",
            f"{result['num_runs']}",
        ]
        for cluster in cluster_names:
            acc = result["per_cluster_accuracy"].get(cluster)
            row.append(f"{acc:.4f}" if acc is not None else "-")

        table.add_row(*row)

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    cluster_names = sorted({cluster for result in results for cluster in result["per_cluster_accuracy"].keys()})
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "embeddings_file": result["embeddings_file"],
            "method": result["method"],
            "accuracy": result["accuracy"],
            "accuracy_std": result["accuracy_std"],
            "num_runs": result["num_runs"],
        }
        for cluster in cluster_names:
            row[f"cluster_{cluster}"] = result["per_cluster_accuracy"].get(cluster)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


# pylint: disable=too-many-locals
def evaluate_newt_single(embeddings_path: str, labels_df: pl.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    sample_ids, all_features = load_embeddings(embeddings_path)
    emb_df = pl.DataFrame({"id": sample_ids, "embedding": all_features.tolist()})

    joined = labels_df.join(emb_df, on="id", how="inner").sort("index")
    if joined.height < labels_df.height:
        logger.warning(f"Join dropped {labels_df.height - joined.height} samples (missing embeddings)")

    all_features = np.array(joined.get_column("embedding").to_list(), dtype=np.float32)
    logger.info(f"Loaded {all_features.shape[0]} samples with {all_features.shape[1]} dimensions")

    # Global L2 normalization
    all_features = l2_normalize(all_features)
    joined = joined.with_columns(pl.Series("embedding", all_features.tolist()))

    tasks = joined.get_column("task").unique().to_list()
    logger.info(f"Found {len(tasks)} tasks")

    scores: list[float] = []
    cluster_scores: dict[str, list[float]] = {}
    for run in range(args.runs):
        run_seed = args.seed + run

        y_preds_all: list[npt.NDArray[np.int_]] = []
        y_trues_all: list[npt.NDArray[np.int_]] = []
        cluster_preds: dict[str, list[npt.NDArray[np.int_]]] = {}
        cluster_trues: dict[str, list[npt.NDArray[np.int_]]] = {}
        for task_name in tasks:
            tdf = joined.filter(pl.col("task") == task_name)

            features = np.array(tdf.get_column("embedding").to_list(), dtype=np.float32)

            labels = tdf.get_column("label").to_numpy()
            is_train = (tdf.get_column("split") == "train").to_numpy()
            cluster = tdf.item(0, "task_cluster")

            x_train = features[is_train]
            y_train = labels[is_train]
            x_test = features[~is_train]
            y_test = labels[~is_train]

            if x_train.size == 0 or x_test.size == 0:
                logger.warning(f"Skipping task {task_name}: empty train or test split")
                continue

            # Per-task centering and L2 normalization
            x_train, x_test = normalize_features(x_train, x_test)

            # Train and evaluate SVM
            y_pred, y_true = evaluate_svm(
                x_train,
                y_train,
                x_test,
                y_test,
                n_iter=args.n_iter,
                n_jobs=args.n_jobs,
                seed=run_seed,
            )

            y_preds_all.append(y_pred)
            y_trues_all.append(y_true)

            # Track per-cluster predictions
            if cluster not in cluster_preds:
                cluster_preds[cluster] = []
                cluster_trues[cluster] = []
            cluster_preds[cluster].append(y_pred)
            cluster_trues[cluster].append(y_true)

        # Micro-averaged accuracy
        y_preds = np.concatenate(y_preds_all)
        y_trues = np.concatenate(y_trues_all)
        acc = float(np.mean(y_preds == y_trues))
        scores.append(acc)
        logger.info(f"Run {run + 1}/{args.runs} - Accuracy: {acc:.4f}")

        # Compute per-cluster accuracy for this run
        for cluster, preds_list in cluster_preds.items():
            preds = np.concatenate(preds_list)
            trues = np.concatenate(cluster_trues[cluster])
            cluster_acc = float(np.mean(preds == trues))
            if cluster not in cluster_scores:
                cluster_scores[cluster] = []
            cluster_scores[cluster].append(cluster_acc)

    # Average per-cluster accuracy across runs
    per_cluster_accuracy: dict[str, float] = {}
    for cluster, accs in cluster_scores.items():
        per_cluster_accuracy[cluster.lower()] = float(np.mean(accs))

    scores_arr = np.array(scores)
    mean_acc = float(scores_arr.mean())
    std_acc = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0

    logger.info(f"Mean accuracy over {args.runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")
    for cluster, acc in sorted(per_cluster_accuracy.items()):
        logger.info(f"  {cluster}: {acc:.4f}")

    return {
        "method": "svm",
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "num_runs": args.runs,
        "per_cluster_accuracy": per_cluster_accuracy,
        "embeddings_file": str(embeddings_path),
    }


def evaluate_newt(args: argparse.Namespace) -> None:
    tic = time.time()

    logger.info(f"Loading NeWT dataset from {args.dataset_path}")
    dataset = NeWT(args.dataset_path)
    labels_path = dataset.labels_path
    logger.info(f"Loading labels from {labels_path}")
    labels_df = pl.read_csv(labels_path).with_row_index(name="index").with_columns(pl.col("id").cast(pl.Utf8))

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        result = evaluate_newt_single(embeddings_path, labels_df, args)
        results.append(result)

    _print_summary_table(results)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("newt.csv")
        _write_results_csv(results, output_path)

    toc = time.time()
    logger.info(f"NeWT benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "newt",
        allow_abbrev=False,
        help="run NeWT benchmark - 164 binary classification tasks evaluated with SVM",
        description="run NeWT benchmark - 164 binary classification tasks evaluated with SVM",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval newt --embeddings "
            "results/hieradet_d_small_dino-v2_0_224px_crop1.0_36032_output.parquet "
            "--dataset-path ~/Datasets/NeWT --dry-run\n"
            "python -m birder.eval newt --embeddings results/newt/*.parquet "
            "--dataset-path ~/Datasets/NeWT\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings/logits parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to NeWT dataset root")
    subparser.add_argument("--runs", type=int, default=3, help="number of evaluation runs")
    subparser.add_argument("--n-iter", type=int, default=100, help="SVM hyperparameter search iterations")
    subparser.add_argument("--n-jobs", type=int, default=8, help="parallel jobs for RandomizedSearchCV")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument(
        "--dir", type=str, default="newt", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_newt(args)
