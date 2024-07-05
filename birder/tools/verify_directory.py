import argparse
import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm

from birder.common import cli
from birder.conf import settings


def verify_directory(args: argparse.Namespace) -> None:
    _ = settings.LOGGING
    batch_size = 32
    transform = v2.Compose(
        [
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    dataset = ImageFolder(args.data_path, transform=transform, loader=read_image)
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
                for idx, (_, _) in enumerate(data_loader):
                    progress.update(batch_size)

            except (OSError, RuntimeError) as e:
                logging.warning(
                    f"File at batch no. {idx} (batch size = {batch_size}, "
                    f"{(idx-1) * batch_size + args.start}) failed to load {e}"
                )

        else:
            for img_path, _ in dataset.samples:
                try:
                    img = dataset.loader(img_path)
                    img = transform(img)
                    if img.size(0) != 3:
                        logging.warning(f"File {img_path} failed to load {img.size()}")

                except (OSError, RuntimeError) as e:
                    logging.warning(f"File {img_path} failed to load {e}")

                progress.update(1)

    logging.info("Done !")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "verify-directory",
        help="loads every image in the dataset and raises exception on corrupted files",
        description="loads every image in the dataset and raises exception on corrupted files",
        epilog=(
            "Usage examples:\n"
            "python tool.py verify-directory --fast data/testing\n"
            "python tool.py verify-directory ~/Datasets/birdsnap\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--fast", default=False, action="store_true", help="use parallel dataloader")
    subparser.add_argument("--start", type=int, default=0, help="start at sample number (skip the beginning)")
    subparser.add_argument("data_path", help="image directory path")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    verify_directory(args)