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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.datahub.evaluation import NeWT
from birder.eval._embeddings import l2_normalize_
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.simpleshot import normalize_features
from birder.eval.methods.svm import evaluate_svm

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NeWTTaskData:
    task_name: str
    cluster: str
    test_ids: list[str]
    x_train: npt.NDArray[np.float32]
    y_train: npt.NDArray[np.int_]
    x_test: npt.NDArray[np.float32]
    y_test: npt.NDArray[np.int_]


def _results_filename(args: argparse.Namespace) -> str:
    return f"newt_svm_runs{args.runs}_niter{args.n_iter}.csv"


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

    for result in sorted(results, key=lambda result: Path(result["embeddings_file"]).name):
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


def _result_to_row(result: dict[str, Any], cluster_names: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "embeddings_file": result["embeddings_file"],
        "method": result["method"],
        "accuracy": result["accuracy"],
        "accuracy_std": result["accuracy_std"],
        "num_runs": result["num_runs"],
    }
    for cluster in cluster_names:
        row[f"cluster_{cluster}"] = result["per_cluster_accuracy"].get(cluster)

    return row


def _append_result_csv(result: dict[str, Any], cluster_names: list[str], output_path: Path) -> None:
    row_df = pl.DataFrame([_result_to_row(result, cluster_names)])
    file_exists = output_path.exists()
    mode = "a" if file_exists is True else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        row_df.write_csv(handle, include_header=file_exists is False)

    logger.info(f"Results updated at {output_path}")


def _row_to_result(row: dict[str, Any], cluster_names: list[str]) -> dict[str, Any]:
    per_cluster_accuracy = {
        cluster: float(value) for cluster in cluster_names if (value := row.get(f"cluster_{cluster}")) is not None
    }

    return {
        "embeddings_file": str(row["embeddings_file"]),
        "method": str(row["method"]),
        "accuracy": float(row["accuracy"]),
        "accuracy_std": float(row["accuracy_std"]),
        "num_runs": int(row["num_runs"]),
        "per_cluster_accuracy": per_cluster_accuracy,
    }


def _load_cached_result(
    existing_df: Optional[pl.DataFrame], embeddings_path: str, cluster_names: list[str]
) -> Optional[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return None

    match_df = existing_df.filter(pl.col("embeddings_file") == embeddings_path)
    if match_df.is_empty() is True:
        return None

    return _row_to_result(match_df.row(0, named=True), cluster_names)


def _summary_results(
    existing_df: Optional[pl.DataFrame], current_results: list[dict[str, Any]], cluster_names: list[str]
) -> list[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return current_results

    return [_row_to_result(row, cluster_names) for row in existing_df.iter_rows(named=True)]


def _prepare_newt_task_data(embeddings_path: str, labels_df: pl.DataFrame) -> list[NeWTTaskData]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = load_embeddings(embeddings_path)

    joined = labels_df.join(emb_df, on="id", how="inner").sort("index")
    if joined.height < labels_df.height:
        logger.warning(f"Join dropped {labels_df.height - joined.height} samples (missing embeddings)")

    all_features = joined.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    logger.info(f"Loaded {all_features.shape[0]} samples with {all_features.shape[1]} dimensions")

    # Global L2 normalization
    l2_normalize_(all_features)
    all_labels = joined.get_column("label").to_numpy().astype(np.int_)
    all_splits = joined.get_column("split").to_numpy()
    all_tasks = joined.get_column("task").to_numpy()
    all_clusters = joined.get_column("task_cluster").to_numpy()
    all_ids = joined.get_column("id").to_numpy()

    task_names = np.unique(all_tasks)
    logger.info(f"Found {len(task_names)} tasks")

    task_data: list[NeWTTaskData] = []
    for task_name in task_names:
        task_mask = all_tasks == task_name
        task_features = all_features[task_mask]
        task_labels = all_labels[task_mask]
        task_splits = all_splits[task_mask]
        task_ids = all_ids[task_mask]
        is_train = task_splits == "train"
        cluster = str(all_clusters[task_mask][0])

        x_train = task_features[is_train]
        y_train = task_labels[is_train]
        x_test = task_features[~is_train]
        y_test = task_labels[~is_train]
        test_ids = [str(sample_id) for sample_id in task_ids[~is_train]]

        if x_train.size == 0 or x_test.size == 0:
            logger.warning(f"Skipping task {task_name}: empty train or test split")
            continue

        # Per-task centering and L2 normalization
        x_train, x_test = normalize_features(x_train, x_test)
        task_data.append(NeWTTaskData(str(task_name), cluster, test_ids, x_train, y_train, x_test, y_test))

    return task_data


def _predict_newt_rows(task_data: list[NeWTTaskData], n_iter: int, n_jobs: int, seed: int) -> pl.DataFrame:
    ids: list[str] = []
    tasks: list[str] = []
    clusters: list[str] = []
    preds: list[npt.NDArray[np.int_]] = []
    trues: list[npt.NDArray[np.int_]] = []

    for td in task_data:
        y_pred, y_true = evaluate_svm(
            td.x_train, td.y_train, td.x_test, td.y_test, n_iter=n_iter, n_jobs=n_jobs, seed=seed
        )
        n = len(td.test_ids)
        ids.extend(td.test_ids)
        tasks.extend([td.task_name] * n)
        clusters.extend([td.cluster] * n)
        preds.append(y_pred)
        trues.append(y_true)

    return pl.DataFrame(
        {
            "id": ids,
            "task": tasks,
            "task_cluster": clusters,
            "y_pred": np.concatenate(preds),
            "y_true": np.concatenate(trues),
        }
    ).with_columns(is_error=(pl.col("y_pred") != pl.col("y_true")))


def predict_newt_single(
    embeddings_path: str, labels_df: pl.DataFrame, n_iter: int, n_jobs: int, seed: int
) -> pl.DataFrame:
    task_data = _prepare_newt_task_data(embeddings_path, labels_df)
    return _predict_newt_rows(task_data, n_iter=n_iter, n_jobs=n_jobs, seed=seed)


def evaluate_newt_single(
    embeddings_path: str, labels_df: pl.DataFrame, runs: int, n_iter: int, n_jobs: int, seed: int
) -> dict[str, Any]:
    task_data = _prepare_newt_task_data(embeddings_path, labels_df)

    scores: list[float] = []
    cluster_scores: dict[str, list[float]] = {}
    for run in range(runs):
        run_seed = seed + run

        pred_df = _predict_newt_rows(task_data, n_iter=n_iter, n_jobs=n_jobs, seed=run_seed)

        # Micro-averaged accuracy
        acc = float(pred_df.select((pl.col("y_pred") == pl.col("y_true")).cast(pl.Float64).mean()).item())
        scores.append(acc)
        logger.info(f"Run {run + 1}/{runs} - Accuracy: {acc:.4f}")

        # Compute per-cluster accuracy for this run
        cluster_acc_df = pred_df.group_by("task_cluster").agg(
            (pl.col("y_pred") == pl.col("y_true")).cast(pl.Float64).mean().alias("accuracy")
        )
        for row in cluster_acc_df.to_dicts():
            cluster = str(row["task_cluster"]).lower()
            cluster_acc = float(row["accuracy"])
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

    logger.info(f"Mean accuracy over {runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")
    for cluster, acc in sorted(per_cluster_accuracy.items()):
        logger.info(f"  {cluster}: {acc:.4f}")

    return {
        "method": "svm",
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "num_runs": runs,
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
    cluster_names = sorted(
        set(labels_df.get_column("task_cluster").cast(pl.Utf8).str.to_lowercase().unique().to_list())
    )
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
        summary_results = _summary_results(existing_df, [], cluster_names)
        if len(summary_results) == 0:
            raise cli.ValidationError("--embeddings is required when no cached results are available")

        logger.info("No embeddings provided, showing cached results")
        _print_summary_table(summary_results)
        toc = time.time()
        logger.info(f"NeWT benchmark completed in {lib.format_duration(toc - tic)}")
        return

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")

        cached_result = _load_cached_result(existing_df, embeddings_path, cluster_names)
        if cached_result is not None:
            logger.info(f"Using cached result for {embeddings_path}")
            results.append(cached_result)
            continue

        if Path(embeddings_path).exists() is False:
            logger.warning(f"Embeddings file not found, skipping: {embeddings_path}")
            continue

        result = evaluate_newt_single(
            embeddings_path, labels_df, runs=args.runs, n_iter=args.n_iter, n_jobs=args.n_jobs, seed=args.seed
        )
        results.append(result)
        if output_path is not None:
            _append_result_csv(result, cluster_names, output_path)
            row_df = pl.DataFrame([_result_to_row(result, cluster_names)])
            if existing_df is None:
                existing_df = row_df
            else:
                existing_df = pl.concat([existing_df, row_df], how="diagonal_relaxed")

    _print_summary_table(_summary_results(existing_df, results, cluster_names))

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
    if args.dataset_path is None:
        raise cli.ValidationError("--dataset-path is required")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_newt(args)
