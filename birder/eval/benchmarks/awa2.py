"""
AwA2 benchmark using MLP probe for multi-label attribute prediction

Paper "Zero-Shot Learning -- A Comprehensive Evaluation of the Good, the Bad and the Ugly"
https://arxiv.org/abs/1707.00600
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
from torchvision.datasets import ImageFolder

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.datahub.evaluation import AwA2
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.mlp import evaluate_mlp
from birder.eval.methods.mlp import train_mlp

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("AwA2 (MLP)", style="dim")
    table.add_column("Macro F1", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Runs", justify="right")

    for result in results:
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['macro_f1']:.4f}",
            f"{result['macro_f1_std']:.4f}",
            f"{result['num_runs']}",
        )

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], attribute_names: list[str], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "embeddings_file": result["embeddings_file"],
            "method": result["method"],
            "metric_mode": result["metric_mode"],
            "macro_f1": result["macro_f1"],
            "macro_f1_std": result["macro_f1_std"],
            "num_runs": result["num_runs"],
        }
        for attr in attribute_names:
            row[f"f1_{attr}"] = result["per_attribute_f1"].get(attr)

        rows.append(row)

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_awa2_metadata(dataset: AwA2) -> tuple[pl.DataFrame, npt.NDArray[np.float32], list[str], list[str], list[str]]:
    """
    Load AwA2 metadata: image paths, class assignments, and attribute matrix.

    Returns
    -------
    metadata_df
        DataFrame with columns: id (image stem), class_name
    attribute_matrix
        Binary attribute matrix of shape (num_classes, num_attributes)
    class_names
        List of class names in order
    train_classes
        List of training class names
    test_classes
        List of test class names
    """

    # Load class names (1-indexed in file)
    class_names: list[str] = []
    with open(dataset.classes_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            class_names.append(parts[1])

    # Load attribute matrix (one row per class, 85 attributes)
    attribute_matrix = np.loadtxt(dataset.predicate_matrix_binary_path, dtype=np.float32)

    # Load train/test class split
    with open(dataset.trainclasses_path, encoding="utf-8") as f:
        train_classes = [line.strip() for line in f if line.strip()]

    with open(dataset.testclasses_path, encoding="utf-8") as f:
        test_classes = [line.strip() for line in f if line.strip()]

    # Load image paths using ImageFolder
    image_dataset = ImageFolder(str(dataset.images_dir))
    rows: list[dict[str, Any]] = []
    for path, class_idx in image_dataset.samples:
        class_name = image_dataset.classes[class_idx]
        rows.append({"id": Path(path).stem, "class_name": class_name})

    metadata_df = pl.DataFrame(rows)

    return (metadata_df, attribute_matrix, class_names, train_classes, test_classes)


def _load_embeddings_with_labels(
    embeddings_path: str,
    metadata_df: pl.DataFrame,
    attribute_matrix: npt.NDArray[np.float32],
    class_names: list[str],
    train_classes: list[str],
    test_classes: list[str],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    sample_ids, all_features = load_embeddings(embeddings_path)
    emb_df = pl.DataFrame({"id": sample_ids, "embedding": all_features.tolist()})

    # Join embeddings with metadata
    joined = metadata_df.join(emb_df, on="id", how="inner")
    if joined.height < metadata_df.height:
        logger.warning(f"Join dropped {metadata_df.height - joined.height} samples (missing embeddings)")

    # Create class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Split into train/test based on class membership
    train_mask = joined.get_column("class_name").is_in(train_classes)
    test_mask = joined.get_column("class_name").is_in(test_classes)

    train_data = joined.filter(train_mask)
    test_data = joined.filter(test_mask)

    # Extract features
    x_train = np.array(train_data.get_column("embedding").to_list(), dtype=np.float32)
    x_test = np.array(test_data.get_column("embedding").to_list(), dtype=np.float32)

    # Get labels from attribute matrix (class-level attributes)
    train_class_indices = [class_to_idx[name] for name in train_data.get_column("class_name").to_list()]
    test_class_indices = [class_to_idx[name] for name in test_data.get_column("class_name").to_list()]

    y_train = attribute_matrix[train_class_indices]
    y_test = attribute_matrix[test_class_indices]

    logger.info(f"Train: {x_train.shape[0]} samples ({len(train_classes)} classes)")
    logger.info(f"Test: {x_test.shape[0]} samples ({len(test_classes)} classes)")
    logger.info(f"Features: {x_train.shape[1]} dims, Attributes: {attribute_matrix.shape[1]}")

    return (x_train, y_train, x_test, y_test)


def _compute_macro_f1(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_], metric_mode: str
) -> tuple[float, int]:
    if metric_mode == "all":
        num_attrs = y_true.shape[1]
        score = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        return (float(score), num_attrs)

    if metric_mode == "present-only":
        present_attrs = np.where(y_true.sum(axis=0) > 0)[0]
        if len(present_attrs) == 0:
            logger.warning("No positive attributes in y_true, falling back to --metric-mode all")
            score = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
            return (float(score), y_true.shape[1])

        score = f1_score(y_true, y_pred, average="macro", labels=present_attrs, zero_division=0.0)
        return (float(score), int(len(present_attrs)))

    raise ValueError(f"Unsupported metric mode: {metric_mode}")


# pylint: disable=too-many-locals
def evaluate_awa2_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.float32],
    attribute_names: list[str],
    args: argparse.Namespace,
    embeddings_path: str,
    device: torch.device,
) -> dict[str, Any]:
    num_attributes = len(attribute_names)

    scores: list[float] = []
    per_attribute_f1_runs: list[dict[str, float]] = []

    for run in range(args.runs):
        run_seed = args.seed + run
        logger.info(f"Run {run + 1}/{args.runs} (seed={run_seed})")

        # Train MLP
        model = train_mlp(
            x_train,
            y_train,
            num_classes=num_attributes,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=run_seed,
        )

        # Evaluate
        y_pred, y_true, _ = evaluate_mlp(model, x_test, y_test, batch_size=args.batch_size, device=device)
        macro_f1, num_attrs_scored = _compute_macro_f1(y_true, y_pred, args.metric_mode)
        scores.append(macro_f1)

        # Per-attribute F1
        per_attribute_f1: dict[str, float] = {}
        for i, attr in enumerate(attribute_names):
            attr_f1 = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0.0)
            per_attribute_f1[attr] = float(attr_f1)

        per_attribute_f1_runs.append(per_attribute_f1)
        logger.info(
            f"Run {run + 1}/{args.runs} - Macro F1 ({args.metric_mode}, {num_attrs_scored} attrs): {macro_f1:.4f}"
        )

    # Average results
    scores_arr = np.array(scores)
    mean_f1 = float(scores_arr.mean())
    std_f1 = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0

    # Average per-attribute F1 across runs
    avg_per_attribute_f1: dict[str, float] = {}
    for attr in attribute_names:
        attr_scores = [run_f1[attr] for run_f1 in per_attribute_f1_runs]
        avg_per_attribute_f1[attr] = float(np.mean(attr_scores))

    logger.info(f"Mean Macro F1 over {args.runs} runs: {mean_f1:.4f} +/- {std_f1:.4f} (std)")

    return {
        "method": "mlp",
        "metric_mode": args.metric_mode,
        "macro_f1": mean_f1,
        "macro_f1_std": std_f1,
        "num_runs": args.runs,
        "per_attribute_f1": avg_per_attribute_f1,
        "embeddings_file": str(embeddings_path),
    }


def evaluate_awa2(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    logger.info(f"Loading AwA2 dataset from {args.dataset_path}")
    logger.info(f"Metric mode: {args.metric_mode}")
    dataset = AwA2(args.dataset_path)
    attribute_names = dataset.attribute_names

    metadata_df, attribute_matrix, class_names, train_classes, test_classes = _load_awa2_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")
    logger.info(f"Train classes: {len(train_classes)}, Test classes: {len(test_classes)}")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_test, y_test = _load_embeddings_with_labels(
            embeddings_path, metadata_df, attribute_matrix, class_names, train_classes, test_classes
        )

        result = evaluate_awa2_single(x_train, y_train, x_test, y_test, attribute_names, args, embeddings_path, device)
        results.append(result)

    _print_summary_table(results)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("awa2.csv")
        _write_results_csv(results, attribute_names, output_path)

    toc = time.time()
    logger.info(f"AwA2 benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "awa2",
        allow_abbrev=False,
        help="run AwA2 benchmark - 85 attribute multi-label classification using MLP probe",
        description="run AwA2 benchmark - 85 attribute multi-label classification using MLP probe",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval awa2 --embeddings "
            "results/awa2_embeddings.parquet "
            "--dataset-path ~/Datasets/Animals_with_Attributes2 --dry-run\n"
            "python -m birder.eval awa2 --embeddings results/awa2_*.parquet "
            "--dataset-path ~/Datasets/Animals_with_Attributes2 --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to AwA2 dataset root")
    subparser.add_argument("--runs", type=int, default=3, help="number of evaluation runs")
    subparser.add_argument("--epochs", type=int, default=100, help="training epochs per run")
    subparser.add_argument("--batch-size", type=int, default=128, help="batch size for training and inference")
    subparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    subparser.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden layer dimension")
    subparser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument(
        "--metric-mode",
        type=str,
        choices=["all", "present-only"],
        default="present-only",
        help="macro F1 mode: all attributes or only attributes present in test split",
    )
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="awa2", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_awa2(args)
