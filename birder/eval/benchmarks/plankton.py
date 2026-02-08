"""
Plankton benchmark using linear probing for phytoplankton classification

Dataset "SYKE-plankton_IFCB_2022", https://b2share.eudat.eu/records/xvnrp-7ga56
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
import torch
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.data.datasets.directory import make_image_dataset
from birder.datahub.evaluation import Plankton
from birder.eval._embeddings import load_embeddings
from birder.eval.methods.linear import evaluate_linear_probe
from birder.eval.methods.linear import train_linear_probe

logger = logging.getLogger(__name__)


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("Plankton (Linear)", style="dim")
    table.add_column("Accuracy", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Runs", justify="right")

    for result in results:
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['accuracy']:.4f}",
            f"{result['accuracy_std']:.4f}",
            f"{result['num_runs']}",
        )

    console.print(table)


def _write_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "embeddings_file": result["embeddings_file"],
                "method": result["method"],
                "accuracy": result["accuracy"],
                "accuracy_std": result["accuracy_std"],
                "num_runs": result["num_runs"],
            }
        )

    pl.DataFrame(rows).write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def _load_plankton_metadata(dataset: Plankton) -> pl.DataFrame:
    """
    Load metadata using make_image_dataset with a fixed class_to_idx

    Returns DataFrame with columns: id (filename stem), label, split
    """

    # Build class_to_idx from train classes only
    class_to_idx = {
        entry.name: idx
        for idx, entry in enumerate(sorted(os.scandir(str(dataset.train_dir)), key=lambda e: e.name))
        if entry.is_dir()
    }

    rows: list[dict[str, Any]] = []
    skipped_no_label = 0
    for split, split_dir in [("train", dataset.train_dir), ("val", dataset.val_dir)]:
        image_dataset = make_image_dataset([str(split_dir)], class_to_idx)
        for i in range(len(image_dataset)):
            path = image_dataset.paths[i].decode("utf-8")
            label = image_dataset.labels[i].item()
            if label == settings.NO_LABEL:
                skipped_no_label += 1
                continue

            rows.append({"id": Path(path).stem, "label": label, "split": split})

    if skipped_no_label > 0:
        logger.info(f"Skipped {skipped_no_label} samples with unknown labels (NO_LABEL)")

    return pl.DataFrame(rows)


def _load_embeddings_with_split(
    embeddings_path: str, metadata_df: pl.DataFrame
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_], npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    emb_df = load_embeddings(embeddings_path)

    train_meta = metadata_df.filter(pl.col("split") == "train").select(["id", "label"])
    val_meta = metadata_df.filter(pl.col("split") == "val").select(["id", "label"])

    train_join = train_meta.join(emb_df, on="id", how="inner")
    val_join = val_meta.join(emb_df, on="id", how="inner")

    dropped_train = train_meta.height - train_join.height
    dropped_val = val_meta.height - val_join.height
    dropped_total = dropped_train + dropped_val
    if dropped_total > 0:
        logger.warning(
            f"Join dropped {dropped_total} samples (missing embeddings): train={dropped_train}, val={dropped_val}"
        )

    x_train = train_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_train = train_join.get_column("label").to_numpy().astype(np.int_)
    x_val = val_join.get_column("embedding").to_numpy().astype(np.float32, copy=False)
    y_val = val_join.get_column("label").to_numpy().astype(np.int_)

    num_classes = y_train.max() + 1
    total_samples = x_train.shape[0] + x_val.shape[0]
    embedding_dim = x_train.shape[1]
    logger.info(f"Loaded {total_samples} samples with {embedding_dim} dimensions, {num_classes} classes")

    logger.info(f"Train: {len(y_train)} samples, Val: {len(y_val)} samples")

    return (x_train, y_train, x_val, y_val)


def evaluate_plankton_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.int_],
    args: argparse.Namespace,
    embeddings_path: str,
    device: torch.device,
) -> dict[str, Any]:
    num_classes = int(y_train.max() + 1)

    scores: list[float] = []
    for run in range(args.runs):
        run_seed = args.seed + run
        logger.info(f"Run {run + 1}/{args.runs} (seed={run_seed})")

        model = train_linear_probe(
            x_train,
            y_train,
            num_classes,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=run_seed,
        )

        y_pred, y_true = evaluate_linear_probe(model, x_val, y_val, batch_size=args.batch_size, device=device)
        acc = float(np.mean(y_pred == y_true))
        scores.append(acc)
        logger.info(f"Run {run + 1}/{args.runs} - Accuracy: {acc:.4f}")

    scores_arr = np.array(scores)
    mean_acc = float(scores_arr.mean())
    std_acc = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0

    logger.info(f"Mean accuracy over {args.runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")

    return {
        "method": "linear",
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "num_runs": args.runs,
        "embeddings_file": str(embeddings_path),
    }


def evaluate_plankton(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    logger.info(f"Loading Plankton dataset from {args.dataset_path}")
    dataset = Plankton(args.dataset_path)
    metadata_df = _load_plankton_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")
        x_train, y_train, x_val, y_val = _load_embeddings_with_split(embeddings_path, metadata_df)

        result = evaluate_plankton_single(x_train, y_train, x_val, y_val, args, embeddings_path, device)
        results.append(result)

    _print_summary_table(results)

    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath("plankton.csv")
        _write_results_csv(results, output_path)

    toc = time.time()
    logger.info(f"Plankton benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "plankton",
        allow_abbrev=False,
        help="run Plankton benchmark - 50 class phytoplankton classification using linear probing",
        description="run Plankton benchmark - 50 class phytoplankton classification using linear probing",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval plankton --embeddings "
            "results/plankton_embeddings.parquet --dataset-path ~/Datasets/plankton --dry-run\n"
            "python -m birder.eval plankton --embeddings results/plankton/*.parquet "
            "--dataset-path ~/Datasets/plankton --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to Plankton dataset root")
    subparser.add_argument("--runs", type=int, default=3, help="number of evaluation runs")
    subparser.add_argument("--epochs", type=int, default=50, help="training epochs per run")
    subparser.add_argument("--batch-size", type=int, default=64, help="batch size for training and inference")
    subparser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="plankton", help="place all outputs in a sub-directory (relative to results)"
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
    evaluate_plankton(args)
