import argparse
import logging
import random
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder

from birder.common import cli
from birder.common import fs_ops
from birder.common import masking
from birder.common import training_utils
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import get_mixup_cutmix
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import reverse_preset
from birder.data.transforms.classification import training_preset

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-branches
def show_iterator(args: argparse.Namespace) -> None:
    reverse_transform = reverse_preset(get_rgb_stats("birder"))
    if args.mode == "training":
        transform = training_preset(
            args.size,
            args.aug_type,
            args.aug_level,
            get_rgb_stats("birder"),
            args.resize_min_scale,
            args.re_prob,
            args.use_grayscale,
            args.ra_num_ops,
            args.ra_magnitude,
            args.augmix_severity,
            args.simple_crop,
        )
    elif args.mode == "inference":
        transform = inference_preset(args.size, get_rgb_stats("birder"), args.center_crop)
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    batch_size = 8
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            (wds_path, dataset_size) = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            (wds_path, dataset_size) = prepare_wds_args(args.data_path, args.wds_size, torch.device("cpu"))

        dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=False,
            transform=transform,
        )
        if args.wds_class_file is None:
            args.wds_class_file = Path(args.data_path).joinpath(settings.CLASS_LIST_NAME)

        class_to_idx = fs_ops.read_class_file(args.wds_class_file)

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
            ax.set_title(f"Original, aug type: {args.aug_type}")

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
        t = get_mixup_cutmix(args.mixup_alpha, num_outputs, args.cutmix)

        def collate_fn(batch: Any) -> Any:
            return t(*default_collate(batch))

        if args.wds is True:
            data_loader = make_wds_loader(
                dataset,
                batch_size,
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

        # Masking
        mask_size = (args.size[0] // args.patch_size, args.size[1] // args.patch_size)
        mask_generator: Optional[masking.Masking]
        if args.masking == "uniform":
            mask_generator = masking.UniformMasking(mask_size, args.mask_ratio)
        elif args.masking == "block":
            max_patches = int(args.mask_ratio * mask_size[0] * mask_size[1])
            mask_generator = masking.BlockMasking(mask_size, 4, max_patches, 0.33, 3.33)
        else:
            mask_generator = None

        for k, (inputs, _) in enumerate(data_loader):
            if k >= no_iterations:
                break

            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

            if mask_generator is not None:
                masks = mask_generator(batch_size)
                inputs = masking.mask_tensor(inputs, masks, patch_factor=args.patch_size)

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
            "python -m birder.tools show-iterator --mode training --aug-level 3\n"
            "python -m birder.tools show-iterator --mode training --size 224 --aug-level 2 --batch\n"
            "python -m birder.tools show-iterator --mode inference --size 320\n"
            "python -m birder.tools show-iterator --mode training --size 224 --batch --wds "
            "--wds-class-file ~/Datasets/imagenet-1k-wds/classes.txt --wds-size 50000 "
            "--data-path ~/Datasets/imagenet-1k-wds/validation\n"
            "python -m birder.tools show-iterator --mode training --batch --size 224 --aug-level 6 --masking uniform\n"
            "python -m birder.tools show-iterator --mode training --size 224 --batch --wds "
            "--data-path data/training_packed\n"
            "python -m birder.tools show-iterator --mode training --batch --mixup-alpha 0.8 --cutmix "
            "--aug-level 8 --data-path ~/Datasets/inat2021/train\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--mode", type=str, choices=["training", "inference"], default="training", help="iterator mode"
    )
    subparser.add_argument("--size", type=int, nargs="+", default=[224], metavar=("H", "W"), help="image size")
    training_utils.add_aug_args(subparser)
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio during inference")
    subparser.add_argument(
        "--batch", default=False, action="store_true", help="show a batch instead of a single sample"
    )
    subparser.add_argument("--mixup-alpha", type=float, help="mixup alpha")
    subparser.add_argument("--cutmix", default=False, action="store_true", help="enable cutmix")
    subparser.add_argument(
        "--masking", type=str, choices=["none", "uniform", "block"], default="none", help="enable masking"
    )
    subparser.add_argument("--mask-ratio", type=float, default=0.5, help="mask ratio")
    subparser.add_argument("--patch-size", type=int, default=16, help="mask base patch size")
    subparser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="image directory path"
    )
    subparser.add_argument("--wds", default=False, action="store_true", help="use webdataset")
    subparser.add_argument("--wds-info", type=str, metavar="FILE", help="wds info file path")
    subparser.add_argument("--wds-class-file", type=str, metavar="FILE", help="class list file")
    subparser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds directory")
    subparser.add_argument(
        "--wds-split", type=str, default="training", metavar="NAME", help="wds dataset split to load"
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.wds is False or args.batch is True, "WDS only works in batch mode"

    args.size = cli.parse_size(args.size)
    show_iterator(args)
