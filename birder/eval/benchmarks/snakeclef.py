"""
SnakeCLEF2023 benchmark using linear probing for observation-level snake species classification

Link: https://www.imageclef.org/SnakeCLEF2023
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.common import lib
from birder.conf import settings
from birder.datahub.evaluation import SnakeCLEF2023
from birder.eval.methods.linear import evaluate_linear_probe
from birder.eval.methods.linear import train_linear_probe

logger = logging.getLogger(__name__)


def _format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _results_filename(args: argparse.Namespace) -> str:
    return (
        f"snakeclef_linear_runs{args.runs}"
        f"_e{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{_format_float_for_filename(args.lr)}"
        ".csv"
    )


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("SnakeCLEF2023 (Linear)", style="dim")
    table.add_column("Accuracy", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Runs", justify="right")

    for result in sorted(results, key=lambda result: Path(result["embeddings_file"]).name):
        table.add_row(
            Path(result["embeddings_file"]).name,
            f"{result['accuracy']:.4f}",
            f"{result['accuracy_std']:.4f}",
            f"{result['num_runs']}",
        )

    console.print(table)


def _result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "embeddings_file": result["embeddings_file"],
        "method": result["method"],
        "accuracy": result["accuracy"],
        "accuracy_std": result["accuracy_std"],
        "num_runs": result["num_runs"],
        "train_observations": result["train_observations"],
        "val_observations": result["val_observations"],
    }


def _append_result_csv(result: dict[str, Any], output_path: Path) -> None:
    row_df = pl.DataFrame([_result_to_row(result)])
    file_exists = output_path.exists()
    mode = "a" if file_exists is True else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        row_df.write_csv(handle, include_header=file_exists is False)

    logger.info(f"Results updated at {output_path}")


def _row_to_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "embeddings_file": str(row["embeddings_file"]),
        "method": str(row["method"]),
        "accuracy": float(row["accuracy"]),
        "accuracy_std": float(row["accuracy_std"]),
        "num_runs": int(row["num_runs"]),
        "train_observations": int(row["train_observations"]),
        "val_observations": int(row["val_observations"]),
    }


def _load_cached_result(existing_df: Optional[pl.DataFrame], embeddings_path: str) -> Optional[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return None

    match_df = existing_df.filter(pl.col("embeddings_file") == embeddings_path)
    if match_df.is_empty() is True:
        return None

    return _row_to_result(match_df.row(0, named=True))


def _summary_results(
    existing_df: Optional[pl.DataFrame], current_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if existing_df is None or existing_df.is_empty() is True:
        return current_results

    return [_row_to_result(row) for row in existing_df.iter_rows(named=True)]


def _normalize_path_str(path: str) -> str:
    return path.replace("\\", "/")


def _sample_to_image_key(sample: str, images_dir: Path) -> str:
    sample = _normalize_path_str(sample)
    images_dir_str = _normalize_path_str(str(images_dir)).rstrip("/")
    prefix = f"{images_dir_str}/"
    if sample.startswith(prefix) is True:
        return sample[len(prefix) :]

    marker = f"/{images_dir.name}/"
    if marker in sample:
        return sample.split(marker, maxsplit=1)[1]

    return sample


def _load_snakeclef_metadata(dataset: SnakeCLEF2023) -> pl.DataFrame:
    train_frames = [
        pl.read_csv(dataset.train_metadata_path).with_columns(pl.lit("train").alias("split")),
    ]

    hmp_dir = dataset.images_dir.joinpath("HMP")
    if hmp_dir.exists() is True and dataset.train_hm_metadata_path.exists() is True:
        logger.info(f"Including rare-class train-hm split from {hmp_dir}")
        train_frames.append(pl.read_csv(dataset.train_hm_metadata_path).with_columns(pl.lit("train").alias("split")))
    elif dataset.train_hm_metadata_path.exists() is True:
        logger.info("HMP images directory not found, skipping train-hm split")

    train_df = (
        pl.concat(train_frames, how="diagonal_relaxed")
        .with_columns(
            pl.col("observation_id").cast(pl.Utf8),
            pl.col("image_path").str.replace_all("\\", "/", literal=True).alias("image_key"),
            pl.col("class_id").alias("label"),
        )
        .select(["image_key", "observation_id", "label", "split"])
    )

    val_df = (
        pl.read_csv(dataset.val_metadata_path)
        .with_columns(
            pl.col("observation_id").cast(pl.Utf8),
            pl.col("image_path").str.replace_all("\\", "/", literal=True).alias("image_key"),
            pl.col("class_id").alias("label"),
            pl.lit("val").alias("split"),
        )
        .select(["image_key", "observation_id", "label", "split"])
    )

    metadata_df = pl.concat([train_df, val_df], how="diagonal_relaxed")

    # Linear probing expects class indices to be contiguous.
    unique_labels = metadata_df.get_column("label").unique().sort()
    label_map = {label: idx for idx, label in enumerate(unique_labels.to_list())}

    return metadata_df.with_columns(pl.col("label").replace_strict(label_map).alias("label"))


def _load_snakeclef_embeddings(embeddings_path: str, images_dir: Path) -> pl.DataFrame:
    logger.info(f"Loading embeddings from {embeddings_path}")

    # SnakeCLEF image filename stems are not globally unique, so join on relative image path
    emb_df = pl.read_parquet(embeddings_path).select(
        pl.col("sample")
        .map_elements(lambda sample: _sample_to_image_key(sample, images_dir), return_dtype=pl.Utf8)
        .alias("image_key"),
        pl.col("embedding"),
    )

    duplicate_keys = emb_df.group_by("image_key").len().filter(pl.col("len") > 1).height
    if duplicate_keys > 0:
        logger.warning(f"Found {duplicate_keys} duplicate image keys in embeddings parquet")

    return emb_df


def _aggregate_observations(
    joined_df: pl.DataFrame,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    if joined_df.is_empty() is True:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int_),
        )

    label_conflicts = joined_df.group_by("observation_id").agg(pl.col("label").n_unique().alias("n_labels"))
    label_conflicts = label_conflicts.filter(pl.col("n_labels") > 1)
    if label_conflicts.is_empty() is False:
        raise RuntimeError(f"Found {label_conflicts.height} observations with conflicting labels")

    observation_ids = joined_df.get_column("observation_id").to_numpy()
    labels = joined_df.get_column("label").to_numpy().astype(np.int_, copy=False)
    embeddings = joined_df.get_column("embedding").to_numpy().astype(np.float32, copy=False)

    unique_obs, first_indices, inverse = np.unique(observation_ids, return_index=True, return_inverse=True)
    obs_embeddings = np.zeros((len(unique_obs), embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_obs), dtype=np.int32)
    np.add.at(obs_embeddings, inverse, embeddings)
    np.add.at(counts, inverse, 1)
    obs_embeddings /= counts[:, None]
    obs_labels = labels[first_indices]

    return (obs_embeddings, obs_labels)


def _load_embeddings_with_split(embeddings_path: str, dataset: SnakeCLEF2023, metadata_df: pl.DataFrame) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
    npt.NDArray[np.float32],
    npt.NDArray[np.int_],
]:
    emb_df = _load_snakeclef_embeddings(embeddings_path, dataset.images_dir)

    train_meta = metadata_df.filter(pl.col("split") == "train").select(["image_key", "observation_id", "label"])
    val_meta = metadata_df.filter(pl.col("split") == "val").select(["image_key", "observation_id", "label"])

    train_join = train_meta.join(emb_df, on="image_key", how="inner")
    val_join = val_meta.join(emb_df, on="image_key", how="inner")

    dropped_train = train_meta.height - train_join.height
    dropped_val = val_meta.height - val_join.height
    dropped_total = dropped_train + dropped_val
    if dropped_total > 0:
        logger.warning(
            f"Join dropped {dropped_total} images (missing embeddings): train={dropped_train}, val={dropped_val}"
        )

    x_train, y_train = _aggregate_observations(train_join)
    x_val, y_val = _aggregate_observations(val_join)
    if len(y_train) == 0 or len(y_val) == 0:
        raise RuntimeError("No observation embeddings available after joining metadata with embeddings")

    expected_train_obs = train_meta.get_column("observation_id").n_unique()
    expected_val_obs = val_meta.get_column("observation_id").n_unique()
    dropped_train_obs = expected_train_obs - len(y_train)
    dropped_val_obs = expected_val_obs - len(y_val)
    if dropped_train_obs > 0 or dropped_val_obs > 0:
        logger.warning(
            f"Observation aggregation dropped {dropped_train_obs + dropped_val_obs} observations "
            f"(no remaining image embeddings): train={dropped_train_obs}, val={dropped_val_obs}"
        )

    unseen_val_classes = len(set(np.unique(y_val)).difference(np.unique(y_train)))
    if unseen_val_classes > 0:
        logger.warning(f"Found {unseen_val_classes} validation classes without train observations after join")

    num_classes = len(np.unique(y_train))
    total_observations = x_train.shape[0] + x_val.shape[0]
    embedding_dim = x_train.shape[1]
    logger.info(
        f"Loaded {total_observations} observations with {embedding_dim} dimensions, {num_classes} train classes"
    )
    logger.info(f"Train: {len(y_train)} observations, Val: {len(y_val)} observations")

    return (x_train, y_train, x_val, y_val)


def evaluate_snakeclef_single(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int_],
    x_val: npt.NDArray[np.float32],
    y_val: npt.NDArray[np.int_],
    args: argparse.Namespace,
    embeddings_path: str,
    device: torch.device,
) -> dict[str, Any]:
    num_classes = int(max(y_train.max(), y_val.max()) + 1)

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
        logger.info(f"Run {run + 1}/{args.runs} - Observation accuracy: {acc:.4f}")

    scores_arr = np.array(scores)
    mean_acc = float(scores_arr.mean())
    std_acc = float(scores_arr.std(ddof=1)) if len(scores) > 1 else 0.0

    logger.info(f"Mean observation accuracy over {args.runs} runs: {mean_acc:.4f} +/- {std_acc:.4f} (std)")

    return {
        "method": "linear",
        "accuracy": mean_acc,
        "accuracy_std": std_acc,
        "num_runs": args.runs,
        "embeddings_file": str(embeddings_path),
        "train_observations": len(y_train),
        "val_observations": len(y_val),
    }


def evaluate_snakeclef(args: argparse.Namespace) -> None:
    tic = time.time()

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")
    logger.info(f"Loading SnakeCLEF2023 dataset from {args.dataset_path}")
    dataset = SnakeCLEF2023(args.dataset_path)
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
        summary_results = _summary_results(existing_df, [])
        if len(summary_results) == 0:
            raise cli.ValidationError("--embeddings is required when no cached results are available")

        logger.info("No embeddings provided, showing cached results")
        _print_summary_table(summary_results)
        toc = time.time()
        logger.info(f"SnakeCLEF2023 benchmark completed in {lib.format_duration(toc - tic)}")
        return

    metadata_df = _load_snakeclef_metadata(dataset)
    logger.info(f"Loaded metadata for {metadata_df.height} images")

    results: list[dict[str, Any]] = []
    total = len(args.embeddings)
    for idx, embeddings_path in enumerate(args.embeddings, start=1):
        logger.info(f"Processing embeddings {idx}/{total}: {embeddings_path}")

        cached_result = _load_cached_result(existing_df, embeddings_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {embeddings_path}")
            results.append(cached_result)
            continue

        if Path(embeddings_path).exists() is False:
            logger.warning(f"Embeddings file not found, skipping: {embeddings_path}")
            continue

        x_train, y_train, x_val, y_val = _load_embeddings_with_split(embeddings_path, dataset, metadata_df)

        result = evaluate_snakeclef_single(x_train, y_train, x_val, y_val, args, embeddings_path, device)
        results.append(result)
        if output_path is not None:
            _append_result_csv(result, output_path)
            row_df = pl.DataFrame([_result_to_row(result)])
            if existing_df is None:
                existing_df = row_df
            else:
                existing_df = pl.concat([existing_df, row_df], how="diagonal_relaxed")

    _print_summary_table(_summary_results(existing_df, results))

    toc = time.time()
    logger.info(f"SnakeCLEF2023 benchmark completed in {lib.format_duration(toc - tic)}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "snakeclef",
        allow_abbrev=False,
        help="run SnakeCLEF2023 benchmark - observation-level snake species classification using linear probing",
        description="run SnakeCLEF2023 benchmark - observation-level snake species classification using linear probing",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval snakeclef --embeddings "
            "results/snakeclef_embeddings.parquet --dataset-path ~/Datasets/SnakeCLEF2023 --dry-run\n"
            "python -m birder.eval snakeclef --embeddings results/snakeclef/*.parquet "
            "--dataset-path ~/Datasets/SnakeCLEF2023 --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--embeddings", type=str, nargs="+", metavar="FILE", help="paths to embeddings parquet files"
    )
    subparser.add_argument("--dataset-path", type=str, metavar="PATH", help="path to SnakeCLEF2023 dataset root")
    subparser.add_argument("--runs", type=int, default=3, help="number of evaluation runs")
    subparser.add_argument("--epochs", type=int, default=50, help="training epochs per run")
    subparser.add_argument("--batch-size", type=int, default=512, help="batch size for training and inference")
    subparser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    subparser.add_argument("--seed", type=int, default=0, help="base random seed")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--dir", type=str, default="snakeclef", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset_path is None:
        raise cli.ValidationError("--dataset-path is required")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_snakeclef(args)
