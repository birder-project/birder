import argparse
import logging
from typing import Any
from typing import get_args

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm

from birder.common import cli
from birder.conf import settings
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader

logger = logging.getLogger(__name__)


def verify_directory(args: argparse.Namespace) -> None:
    batch_size = 32
    has_errors = False
    num_verified = 0
    num_valid = 0
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    for data_path in args.data_path:
        dataset = ImageFolder(data_path, transform=transform, loader=get_image_loader(args.img_loader, args.channels))
        total = len(dataset)
        dataset.samples = dataset.samples[args.start :]
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            drop_last=False,
        )
        with tqdm(total=total, initial=args.start, unit="images", unit_scale=True, leave=False) as progress:
            if args.fast is True:
                idx = 0
                try:
                    for idx, (images, _) in enumerate(data_loader):
                        current_batch_size = images.size(0)
                        num_verified += current_batch_size
                        num_valid += current_batch_size
                        progress.update(current_batch_size)

                except (OSError, RuntimeError) as e:
                    logger.warning(
                        f"File at batch no. {idx} (batch size = {batch_size}, "
                        f"{(idx-1) * batch_size + args.start}) failed to load {e}"
                    )
                    raise

            else:
                for img_path, _ in dataset.samples:
                    num_verified += 1
                    try:
                        img = dataset.loader(img_path)
                        img = transform(img)
                        if img.size(0) != args.channels:
                            has_errors = True
                            logger.warning(f"File {img_path} failed to load {img.size()}")
                        else:
                            num_valid += 1

                    except (OSError, RuntimeError) as e:
                        has_errors = True
                        logger.warning(f"File {img_path} failed to load {e}")

                    progress.update(1)

        logger.info(f"Finished {data_path}")

    logger.info(f"Verified {num_verified:,} images, {num_valid:,} valid")
    if has_errors is True:
        raise RuntimeError("Directory verification failed")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "verify-directory",
        allow_abbrev=False,
        help="loads every image in the dataset and raises exception on corrupted files",
        description="loads every image in the dataset and raises exception on corrupted files",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools verify-directory --fast data/testing\n"
            "python -m birder.tools verify-directory ~/Datasets/birdsnap\n"
            "python -m birder.tools verify-directory --fast data/training data/validation data/testing data/raw_data\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--fast", default=False, action="store_true", help="use parallel loading and stop at the first error"
    )
    subparser.add_argument("--start", type=int, default=0, help="start at sample number (skip the beginning)")
    subparser.add_argument(
        "--img-loader",
        type=str,
        choices=get_args(ImageLoaderName),
        default="tv",
        help="backend to load and decode images",
    )
    subparser.add_argument(
        "--channels", type=int, default=settings.DEFAULT_NUM_CHANNELS, metavar="N", help="no. of image channels"
    )
    subparser.add_argument("data_path", nargs="+", help="image directory paths")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    verify_directory(args)
