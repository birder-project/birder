"""
PlantDoc benchmark using SimpleShot for plant disease classification

Paper "PlantDoc: A Dataset for Visual Plant Disease Detection",
https://arxiv.org/abs/1911.10317
"""

import argparse
import logging
import os
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
from birder.data.datasets.directory import make_image_dataset
from birder.datahub.evaluation import PlantDoc
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.simpleshot import evaluate_simpleshot
from birder.eval.methods.simpleshot import sample_k_shot

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]], k_shots: list[int]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("PlantDoc (SimpleShot)", style="dim")
    for k in k_shots:
        table.add_column(f"{k}-shot", justify="right")

    table.add_column("Runs", justify="right")

    for result in results:
        row = [Path(result["embeddings_file"]).name]
        for k in k_shots:
            acc = result["accuracies"].get(k)
            row.append(f"{acc:.4f}" if acc is not None else "-")

        row.append(f"{result['num_runs']}")
        table.add_row(*row)

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], k_shots: list[int], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "embeddings_file": result["embeddings_file"],
            "method": result["method"],
            "num_runs": result["num_runs"],
        }
        for k in k_shots:
            row[f"{k}_shot_acc"] = result["accuracies"].get(k)
            row[f"{k}_shot_std"] = result["accuracies_std"].get(k)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_plantdoc_metadata(dataset: PlantDoc) -> pl.DataFrame:
    """
    Load metadata using make_image_dataset with a fixed class_to_idx

    Returns DataFrame with columns: id (filename stem), label, split
    """

    # Build unified class_to_idx from the union of both splits
    all_classes: set[str] = set()
    for split_dir in [dataset.train_dir, dataset.test_dir]:
        all_classes.update(entry.name for entry in os.scandir(str(split_dir)) if entry.is_dir())

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(all_classes))}

    rows: list[dict[str, Any]] = []
    for split, split_dir in [("train", dataset.train_dir), ("test", dataset.test_dir)]:
        image_dataset = make_image_dataset([str(split_dir)], class_to_idx)
        for i in range(len(image_dataset)):
            path = image_dataset.paths[i].decode("utf-8")
            label = image_dataset.labels[i].item()
            rows.append({"id": Path(path).stem, "label": label, "split": split})

    return pl.DataFrame(rows)


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
    splits = joined.get_column("split").to_list()

    is_train = np.array([s == "train" for s in splits], dtype=bool)
    is_test = np.array([s == "test" for s in splits], dtype=bool)

    num_classes = all_labels.max() + 1
    logger.info(
        f"Loaded {all_features.shape[0]} samples with {all_features.shape[1]} dimensions, {num_classes} classes"
    )

    x_train = all_features[is_train]
    y_train = all_labels[is_train]
    x_test = all_features[is_test]
    y_test = all_labels[is_test]

    logger.info(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")

    return (x_train, y_train, x_test, y_test)


def evaluate_plantdoc_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.int_],
    k_shot: int,
    num_runs: int,
    seed: int,
    embeddings_path: str,
) -> dict[str, Any]:
    logger.info(f"Evaluating {k_shot}-shot")

    scores: list[float] = []
    for run in range(num_runs):
        run_seed = seed + run
        rng = np.random.default_rng(run_seed)

        # Sample k examples per class
        x_train_k_shot, y_train_k_shot = sample_k_shot(x_train, y_train, k_shot, rng)

        # Evaluate using SimpleShot
        y_pred, y_true = evaluate_simpleshot(x_train_k_shot, y_train_k_shot, x_test, y_test)

        acc = float(np.mean(y_pred == y_true))
        scores.append(acc)
        logger.info(f"Run {run + 1}/{num_runs} - Accuracy: {acc:.4f}")

    scores_arr = np.array(scores)
    mean_acc = float(scores_arr.mean())
    std_acc = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0

    logger.info(f"Mean accuracy over {num_runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")

    return {
        "method": "simpleshot",
        "k_shot": k_shot,
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "num_runs": num_runs,
        "embeddings_file": str(embeddings_path),
    }


def evaluate_plantdoc(args: argparse.Namespace) -> None:
    tic = time.time()

    logger.info(f"Loading PlantDoc dataset from {args.dataset_path}")
    dataset = PlantDoc(args.dataset_path)
    metadata_df = _load_plantdoc_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_test, y_test = _load_embeddings_with_split(embeddings_path, metadata_df)

        accuracies: dict[int, float] = {}
        accuracies_std: dict[int, float] = {}
        for k_shot in args.k_shot:
            single_result = evaluate_plantdoc_single(
                x_train, y_train, x_test, y_test, k_shot, args.runs, args.seed, embeddings_path
            )
            accuracies[k_shot] = single_result["accuracy"]
            accuracies_std[k_shot] = single_result["accuracy_std"]

        results.append(
            {
                "embeddings_file": str(embeddings_path),
                "method": "simpleshot",
                "num_runs": args.runs,
                "accuracies": accuracies,
                "accuracies_std": accuracies_std,
            }
        )

    _print_summary_table(results, args.k_shot)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("plantdoc.csv")
        _write_results_csv(results, args.k_shot, output_path)

    toc = time.time()
    logger.info(f"PlantDoc benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "plantdoc",
        allow_abbrev=False,
        help="run PlantDoc benchmark - 27 class plant disease classification using SimpleShot",
        description="run PlantDoc benchmark - 27 class plant disease classification using SimpleShot",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval plantdoc --embeddings "
            "results/plantdoc_embeddings.parquet --dataset-path ~/Datasets/PlantDoc --dry-run\n"
            "python -m birder.eval plantdoc --embeddings results/plantdoc/*.parquet "
            "--dataset-path ~/Datasets/PlantDoc\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to PlantDoc dataset root")
    subparser.add_argument(
        "--k-shot", type=int, nargs="+", default=[2, 5], help="number of examples per class for few-shot learning"
    )
    subparser.add_argument("--runs", type=int, default=5, help="number of evaluation runs")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument(
        "--dir", type=str, default="plantdoc", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_plantdoc(args)
