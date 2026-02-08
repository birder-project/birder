"""
PlantNet-300K benchmark using SimpleShot for plant species classification

Paper "Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution"
https://openreview.net/forum?id=eLYinD0TtIt
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
from birder.datahub.evaluation import PlantNet
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.simpleshot import evaluate_simpleshot
from birder.eval.methods.simpleshot import sample_k_shot

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]], k_shots: list[int]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("PlantNet (SimpleShot)", style="dim")
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


def _load_plantnet_metadata(dataset: PlantNet) -> pl.DataFrame:
    """
    Load metadata from ImageFolder structure

    Returns DataFrame with columns: id (filename stem), label, split
    """

    rows: list[dict[str, Any]] = []
    for split, split_dir in [("train", dataset.train_dir), ("val", dataset.val_dir), ("test", dataset.test_dir)]:
        if not split_dir.exists():
            continue

        image_dataset = ImageFolder(str(split_dir))
        for path, label in image_dataset.samples:
            rows.append({"id": Path(path).stem, "label": label, "split": split})

    return pl.DataFrame(rows)


def _load_embeddings_with_split(
    embeddings_path: str, metadata_df: pl.DataFrame
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_], npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = load_embeddings(embeddings_path)

    train_meta = metadata_df.filter(pl.col("split") == "train").select(["id", "label"])
    test_meta = metadata_df.filter(pl.col("split") == "test").select(["id", "label"])

    train_join = train_meta.join(emb_df, on="id", how="inner")
    test_join = test_meta.join(emb_df, on="id", how="inner")

    dropped_train = train_meta.height - train_join.height
    dropped_test = test_meta.height - test_join.height
    dropped_total = dropped_train + dropped_test
    if dropped_total > 0:
        logger.warning(
            f"Join dropped {dropped_total} samples (missing embeddings): train={dropped_train}, test={dropped_test}"
        )

    x_train = train_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_train = train_join.get_column("label").to_numpy().astype(np.int_)
    x_test = test_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_test = test_join.get_column("label").to_numpy().astype(np.int_)

    num_classes = metadata_df.get_column("label").max() + 1  # type: ignore[operator]
    total_samples = x_train.shape[0] + x_test.shape[0]
    embedding_dim = x_train.shape[1]
    logger.info(f"Loaded {total_samples} samples with {embedding_dim} dimensions, {num_classes} classes")

    logger.info(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")

    return (x_train, y_train, x_test, y_test)


def evaluate_plantnet_single(
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


def evaluate_plantnet(args: argparse.Namespace) -> None:
    tic = time.time()

    logger.info(f"Loading PlantNet dataset from {args.dataset_path}")
    dataset = PlantNet(args.dataset_path)
    metadata_df = _load_plantnet_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_test, y_test = _load_embeddings_with_split(embeddings_path, metadata_df)

        accuracies: dict[int, float] = {}
        accuracies_std: dict[int, float] = {}
        for k_shot in args.k_shot:
            single_result = evaluate_plantnet_single(
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
        output_path = output_dir.joinpath("plantnet.csv")
        _write_results_csv(results, args.k_shot, output_path)

    toc = time.time()
    logger.info(f"PlantNet benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "plantnet",
        allow_abbrev=False,
        help="run PlantNet-300K benchmark - 1081 species classification using SimpleShot",
        description="run PlantNet-300K benchmark - 1081 species classification using SimpleShot",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval plantnet --embeddings "
            "results/plantnet_embeddings.parquet --dataset-path ~/Datasets/plantnet_300K --dry-run\n"
            "python -m birder.eval plantnet --embeddings results/plantnet/*.parquet "
            "--dataset-path ~/Datasets/plantnet_300K\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to PlantNet dataset root")
    subparser.add_argument(
        "--k-shot", type=int, nargs="+", default=[2, 5], help="number of examples per class for few-shot learning"
    )
    subparser.add_argument("--runs", type=int, default=3, help="number of evaluation runs")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument(
        "--dir", type=str, default="plantnet", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_plantnet(args)
