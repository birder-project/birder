import argparse
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder

from birder.common import cli
from birder.conf import settings
from birder.core.dataloader.webdataset import make_wds_loader
from birder.core.datasets.webdataset import make_wds_dataset
from birder.core.datasets.webdataset import wds_size
from birder.core.transforms.classification import get_mixup_cutmix
from birder.core.transforms.classification import get_rgb_values
from birder.core.transforms.classification import inference_preset
from birder.core.transforms.classification import reverse_preset
from birder.core.transforms.classification import training_preset


# pylint: disable=too-many-locals,too-many-branches
def show_iterator(args: argparse.Namespace) -> None:
    reverse_transform = reverse_preset(get_rgb_values("calculated"))
    if args.mode == "training":
        transform = training_preset((args.size, args.size), args.aug_level, get_rgb_values("calculated"))
    elif args.mode == "inference":
        transform = inference_preset((args.size, args.size), args.center_crop, get_rgb_values("calculated"))
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    batch_size = 8
    if args.wds is True:
        (wds_path, _) = cli.wds_braces_from_path(Path(args.data_path))
        if args.wds_size is not None:
            dataset_size = args.wds_size

        else:
            dataset_size = wds_size(wds_path, 1)
            logging.info(f"WDS dataset size is {dataset_size:,}")

        dataset = make_wds_dataset(
            wds_path,
            batch_size,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=False,
            transform=transform,
        )
        if args.wds_class_file is None:
            args.wds_class_file = Path(args.data_path).joinpath(settings.CLASS_LIST_NAME)

        class_to_idx = cli.read_class_file(args.wds_class_file)

    else:
        dataset = ImageFolder(args.data_path, transform=transform)
        class_to_idx = dataset.class_to_idx

    no_iterations = 6
    if args.batch is False:
        samples = random.sample(dataset.imgs, no_iterations)
        cols = 4
        rows = 3
        for img_path, _ in samples:
            img = dataset.loader(img_path)
            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

            # Show original
            ax = fig.add_subplot(grid_spec[0, 0:cols])
            ax.imshow(img)
            ax.set_title("Original")

            # Show transformed
            counter = 0
            for i in range(cols):
                for j in range(1, rows):
                    transformed_img = F.to_pil_image(reverse_transform(transform(img)))

                    ax = fig.add_subplot(grid_spec[j, i])
                    ax.imshow(np.asarray(transformed_img))
                    ax.set_title(f"#{counter}")
                    counter += 1

            plt.show()

    else:
        cols = 4
        rows = 2
        num_outputs = len(class_to_idx)
        t = get_mixup_cutmix(0.8, num_outputs, args.cutmix)

        def collate_fn(batch: Any) -> Any:
            return t(*default_collate(batch))

        if args.wds is True:
            data_loader = make_wds_loader(
                dataset,
                batch_size,
                shuffle=False,  # Shuffle is done at the wds dataset
                num_workers=1,
                prefetch_factor=1,
                collate_fn=collate_fn,
                world_size=1,
                pin_memory=False,
            )

        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

        for k, (inputs, _) in enumerate(data_loader):
            if k >= no_iterations:
                break

            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

            # Show transformed
            counter = 0
            for i in range(cols):
                for j in range(rows):
                    img = inputs[i + cols * j]
                    transformed_img = F.to_pil_image(reverse_transform(img))

                    ax = fig.add_subplot(grid_spec[j, i])
                    ax.imshow(np.asarray(transformed_img))
                    ax.set_title(f"#{counter}")
                    counter += 1

            plt.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "show-iterator",
        allow_abbrev=False,
        help="show training / inference iterator output vs input",
        description="show training / inference iterator output vs input",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools show-iterator --mode training --size 224 --aug-level 3\n"
            "python -m birder.tools show-iterator --mode training --size 224 --aug-level 2 --batch\n"
            "python -m birder.tools show-iterator --mode inference --size 320\n"
            "python -m birder.tools show-iterator --mode training --size 224 --batch --wds "
            "--wds-class-file ~/Datasets/imagenet-1k-wds/classes.txt --wds-size 50000 "
            "--data-path ~/Datasets/imagenet-1k-wds/validation\n"
            "python -m birder.tools show-iterator --mode training --size 384 --aug-level 4 --batch "
            "--cutmix --wds --data-path ~/Datasets/imagenet-1k-wds/validation\n"
            "python -m birder.tools show-iterator --mode training --size 224 --batch --wds "
            "--data-path data/training_packed\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--mode", type=str, choices=["training", "inference"], default="training", help="iterator mode"
    )
    subparser.add_argument("--size", type=int, required=True, help="image size")
    subparser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=3,
        help="magnitude of augmentations (0 off -> 4 highest)",
    )
    subparser.add_argument("--center-crop", type=float, default=1.0, help="Center crop ratio during inference")
    subparser.add_argument(
        "--batch", default=False, action="store_true", help="Show a batch instead of a single sample"
    )
    subparser.add_argument("--cutmix", default=False, action="store_true", help="enable cutmix")
    subparser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="image directory path"
    )
    subparser.add_argument("--wds", default=False, action="store_true", help="use webdataset")
    subparser.add_argument("--wds-class-file", type=str, default=None, help="class list file")
    subparser.add_argument("--wds-size", type=int, help="size of the wds directory")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.wds is False or args.batch is True, "WDS only works in batch mode"

    show_iterator(args)
