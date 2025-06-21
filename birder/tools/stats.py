import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from birder.common import cli
from birder.conf import settings

logger = logging.getLogger(__name__)


def detection_object_count(directory: Path) -> tuple[Counter[str], int]:
    """
    Assumes directory is in LabelMe format
    """

    file_count = 0
    detection_count: Counter[str] = Counter()
    for d in directory.iterdir():
        if d.is_dir() is False:
            continue

        for file_path in d.glob("*.json"):
            with open(file_path, "r", encoding="utf=8") as handle:
                raw_label = json.load(handle)

            if raw_label["flags"].get("unknown", False) is True:
                # Don't count images with unknown species
                continue

            for shape in raw_label["shapes"]:
                detection_count.update([shape["label"]])

            file_count += 1

    return (detection_count, file_count)


def directory_label_count(directory: str | Path) -> Counter[str]:
    dataset = ImageFolder(directory)
    labels = [dataset.classes[sample[1]] for sample in dataset.samples]
    return Counter(labels)


def class_graph(args: argparse.Namespace) -> None:
    if args.detection_class_graph is True:
        (label_count, _) = detection_object_count(Path(args.data_path))
    else:
        label_count = directory_label_count(args.data_path)

    total_count = sum(label_count.values())
    (sorted_classes, sorted_count) = list(zip(*label_count.most_common()))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.barh(sorted_classes, sorted_count, color=plt.get_cmap("RdYlGn")(sorted_count))
    for i, count in enumerate(sorted_count):
        ax.text(
            count + 3,
            i,
            str(count),
            color="dimgrey",
            va="center",
            fontsize="small",
            fontweight="light",
            clip_on=True,
        )

    ax.set_title(
        f"{total_count:n} samples, {len(sorted_classes)} classes "
        f"({total_count / len(sorted_classes):.0f} samples per class on average)"
    )

    plt.show()


def set_size(_args: argparse.Namespace) -> None:
    training_dataset = ImageFolder(settings.TRAINING_DATA_PATH)
    validation_dataset = ImageFolder(settings.VALIDATION_DATA_PATH)
    testing_dataset = ImageFolder(settings.TESTING_DATA_PATH)
    logger.info(
        f"Training:   {len(training_dataset):,} samples with {len(training_dataset.classes)} classes "
        f"(average of {round(len(training_dataset) / len(training_dataset.classes))} per class)"
    )
    logger.info(
        f"Validation: {len(validation_dataset):,} samples with {len(validation_dataset.classes)} classes "
        f"(average of {round(len(validation_dataset) / len(validation_dataset.classes))} per class)"
    )
    logger.info(
        f"Testing:    {len(testing_dataset):,} samples with {len(testing_dataset.classes)} classes "
        f"(average of {round(len(testing_dataset) / len(testing_dataset.classes))} per class)"
    )
    logger.info(
        f"Total of {len(training_dataset) + len(validation_dataset) + len(testing_dataset):,} " "classification samples"
    )

    logger.info("---")
    (training_detection_count, training_file_count) = detection_object_count(
        settings.TRAINING_DETECTION_ANNOTATIONS_PATH
    )
    if training_file_count > 0:
        logger.info(
            f"Detection training:   {training_file_count:,} samples containing "
            f"{sum(training_detection_count.values()):,} "
            f"objects with {len(training_detection_count.keys())} classes (average of "
            f"{sum(training_detection_count.values()) / training_file_count:.1f} objects per sample)"
        )

    (validation_detection_count, validation_file_count) = detection_object_count(
        settings.VALIDATION_DETECTION_ANNOTATIONS_PATH
    )
    if validation_file_count > 0:
        logger.info(
            f"Detection validation: {validation_file_count:,} samples containing "
            f"{sum(validation_detection_count.values()):,} "
            f"objects with {len(validation_detection_count.keys())} classes (average of "
            f"{sum(validation_detection_count.values()) / validation_file_count:.1f} objects per sample)"
        )

    logger.info(f"Total of {training_file_count + validation_file_count:,} detection samples")

    logger.info("---")
    weak_dataset = ImageFolder(settings.WEAKLY_LABELED_DATA_PATH)
    logger.info(
        f"Weakly labeled: {len(weak_dataset):,} samples with {len(weak_dataset.classes)} classes "
        f"(average of {round(len(weak_dataset) / len(weak_dataset.classes))} per class)"
    )
    weak_val_dataset = ImageFolder(settings.WEAKLY_VAL_LABELED_DATA_PATH)
    logger.info(
        f"Weakly labeled validation: {len(weak_val_dataset):,} samples with {len(weak_val_dataset.classes)} classes "
        f"(average of {round(len(weak_val_dataset) / len(weak_val_dataset.classes))} per class)"
    )

    logger.info("---")
    num_classes = len(set(training_dataset.class_to_idx.keys()).union(set(weak_dataset.class_to_idx.keys())))
    logger.info(f"Total of {num_classes} unique classes")


def mean_and_std(args: argparse.Namespace) -> None:
    dataset = ImageFolder(
        args.data_path,
        transform=v2.Compose(
            [
                v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        loader=decode_image,
    )
    batch_size = 64
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
    )
    num_samples = len(dataset)
    mean = np.zeros(3, dtype=np.float64)
    mean_squares = np.zeros(3, dtype=np.float64)
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=True) as progress:
        for images, _ in data_loader:
            mean += images.mean(dim=(2, 3)).sum(dim=0).numpy().astype(np.float64) / num_samples
            mean_squares += images.square().mean(dim=(2, 3)).sum(dim=0).numpy().astype(np.float64) / num_samples

            progress.update(n=batch_size)

    std = np.sqrt(mean_squares - np.square(mean))
    rgb_stats = {"mean": mean.round(decimals=4).tolist(), "std": std.round(decimals=4).tolist()}
    logger.info(f"mean: {rgb_stats['mean']}")
    logger.info(f"std: {rgb_stats['std']}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "stats",
        allow_abbrev=False,
        help="show image directory statistics",
        description="show image directory statistics",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools stats --class-graph\n"
            "python -m birder.tools stats --class-graph --data-path data/validation\n"
            "python -m birder.tools stats --detection-class-graph --data-path "
            "data/detection_data/training_annotations\n"
            "python -m birder.tools stats --set-size\n"
            "python -m birder.tools stats --rgb\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--class-graph", default=False, action="store_true", help="show class sample distribution")
    subparser.add_argument(
        "--detection-class-graph",
        default=False,
        action="store_true",
        help="show class sample distribution for detection directory",
    )
    subparser.add_argument("--set-size", default=False, action="store_true", help="show sample count per set")
    subparser.add_argument(
        "--rgb",
        default=False,
        action="store_true",
        help="calculate rgb mean and std",
    )
    subparser.add_argument("--data-path", type=str, default=settings.TRAINING_DATA_PATH, help="image directory")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.class_graph is True or args.detection_class_graph:
        class_graph(args)

    if args.set_size is True:
        set_size(args)

    if args.rgb is True:
        mean_and_std(args)
