import argparse
import logging
import math
import random
from collections.abc import Callable
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any
from typing import Optional
from typing import get_args

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
from birder.common import training_cli
from birder.conf import settings
from birder.data.collators.naflex import NaFlexBatchProcessor
from birder.data.collators.naflex import NaFlexBatchSpec
from birder.data.collators.naflex import NaFlexMixupTrainingCollator
from birder.data.collators.naflex import NaFlexTrainingCollator
from birder.data.collators.naflex import resolve_naflex_batch_specs
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.naflex import NaFlexMultiScaleDataset
from birder.data.datasets.webdataset import get_wds_num_shards
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms import naflex
from birder.data.transforms.classification import get_mixup_cutmix
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import reverse_preset
from birder.data.transforms.classification import training_preset

logger = logging.getLogger(__name__)


def _unpatchify_naflex(
    patches: torch.Tensor, grid_size: torch.Tensor, valid_mask: torch.Tensor, channels: int
) -> torch.Tensor:
    grid_h, grid_w = grid_size.tolist()
    patches = patches[valid_mask]
    patch_size = math.isqrt(patches.size(1) // channels)
    image = patches.reshape(grid_h, grid_w, channels, patch_size, patch_size)

    return image.permute(2, 0, 3, 1, 4).reshape(channels, grid_h * patch_size, grid_w * patch_size)


def _get_mask_generator(args: argparse.Namespace, mask_size: tuple[int, int]) -> Optional[masking.Masking]:
    if args.masking == "uniform":
        return masking.UniformMasking(mask_size, args.mask_ratio, min_mask_size=args.min_mask_size)
    if args.masking == "block":
        max_patches = int(args.mask_ratio * mask_size[0] * mask_size[1])
        return masking.BlockMasking(mask_size, 4, max_patches, 0.33, 3.33)
    if args.masking == "fixed-size-block":
        return masking.FixedSizeBlockMasking(
            mask_size, args.mask_ratio, block_size=3, mask_ratio_adjust=0.07, inverse_mask=True
        )
    if args.masking == "roll-block":
        num_masking_patches = int(args.mask_ratio * mask_size[0] * mask_size[1])
        return masking.RollBlockMasking(mask_size, num_masking_patches=num_masking_patches)
    if args.masking == "inverse-roll":
        num_masking_patches = int(args.mask_ratio * mask_size[0] * mask_size[1])
        return masking.InverseRollBlockMasking(mask_size, num_masking_patches=num_masking_patches)

    return None


def show_iterator(args: argparse.Namespace) -> None:
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    reverse_transform = reverse_preset(rgb_stats)
    naflex_specs: tuple[NaFlexBatchSpec, ...] = ()
    naflex_transforms: Optional[dict[NaFlexBatchSpec, Callable[..., torch.Tensor]]] = None
    if args.naflex is True:
        fixed_naflex_patch_size = args.patch_size
        inference_spec = resolve_naflex_batch_specs(args.size, args.patch_size)[0]
        if args.mode == "training":

            def make_naflex_transform(spec: NaFlexBatchSpec) -> Callable[..., torch.Tensor]:
                return naflex.training_preset(
                    spec.patch_size,
                    spec.max_seq_len,
                    args.aug_type,
                    args.aug_level,
                    rgb_stats,
                    resize_min_scale=args.resize_min_scale,
                    re_prob=args.re_prob,
                    use_grayscale=args.use_grayscale,
                    ra_num_ops=args.ra_num_ops,
                    ra_magnitude=args.ra_magnitude,
                    augmix_severity=args.augmix_severity,
                    clip_color_jitter_prob=args.clip_color_jitter_prob,
                    clip_gray_prob=args.clip_gray_prob,
                )

            naflex_specs = resolve_naflex_batch_specs(
                args.size, args.patch_size, args.naflex_sizes, args.naflex_patch_sizes
            )
            logger.info(f"Resolved NaFlex batch specifications: {list(naflex_specs)}")
            if len(set(naflex_specs)) > 1:
                naflex_transforms = {spec: make_naflex_transform(spec) for spec in naflex_specs}
                transform = None if args.batch is True else naflex_transforms[naflex_specs[0]]
            else:
                spec = naflex_specs[0]
                fixed_naflex_patch_size = spec.patch_size
                transform = make_naflex_transform(spec)
        elif args.mode == "inference":
            transform = naflex.inference_preset(inference_spec.patch_size, inference_spec.max_seq_len, rgb_stats)
        else:
            raise ValueError(f"Unknown mode={args.mode}")
    elif args.mode == "training":
        transform = training_preset(
            args.size,
            args.aug_type,
            args.aug_level,
            rgb_stats,
            resize_min_scale=args.resize_min_scale,
            simple_crop=args.simple_crop,
            re_prob=args.re_prob,
            use_grayscale=args.use_grayscale,
            ra_num_ops=args.ra_num_ops,
            ra_magnitude=args.ra_magnitude,
            augmix_severity=args.augmix_severity,
            clip_color_jitter_prob=args.clip_color_jitter_prob,
            clip_gray_prob=args.clip_gray_prob,
        )
    elif args.mode == "inference":
        transform = inference_preset(args.size, rgb_stats, args.center_crop, args.simple_crop)
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    batch_size = 8
    naflex_batch_processor: Optional[NaFlexBatchProcessor] = None
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            wds_path, dataset_size = prepare_wds_args(args.data_path, args.wds_size, torch.device("cpu"))

        if args.wds_class_file is None:
            args.wds_class_file = Path(args.data_path).joinpath(settings.CLASS_LIST_NAME)

        class_to_idx = fs_ops.read_class_file(args.wds_class_file)

        dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=False,
            transform=transform,
            image_decoder=args.img_loader,
            channels=args.channels,
        )

    else:
        dataset = ImageFolder(
            args.data_path,
            transform=transform,
            loader=get_image_loader(args.img_loader, args.channels),
        )
        class_to_idx = dataset.class_to_idx

    no_iterations = 6
    if args.batch is False:
        assert transform is not None
        samples = random.sample(dataset.imgs, no_iterations)
        cols = 4
        rows = 3
        for img_path, _ in samples:
            img = dataset.loader(img_path)
            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

            # Show original
            ax = fig.add_subplot(grid_spec[0, 0:cols])
            ax.imshow(np.asarray(F.to_pil_image(img)))
            aug_type = f"naflex/{args.aug_type}" if args.naflex is True else args.aug_type
            ax.set_title(f"Original, aug type: {aug_type}")

            # Show transformed
            counter = 0
            for i in range(cols):
                for j in range(1, rows):
                    transformed_img = F.to_pil_image(reverse_transform(transform(img)))

                    ax = fig.add_subplot(grid_spec[j, i])
                    ax.imshow(np.asarray(transformed_img))
                    if args.naflex is True:
                        ax.set_title(f"#{counter}, {transformed_img.height}x{transformed_img.width}")
                    else:
                        ax.set_title(f"#{counter}")
                    counter += 1

            plt.show()

    else:
        cols = 4
        rows = 2
        num_outputs = len(class_to_idx)
        collate_fn: Callable[[Any], Any]
        if args.naflex is True:
            collator_patch_size = None if naflex_transforms is not None else fixed_naflex_patch_size
            if args.mixup_alpha is None:
                naflex_collator = NaFlexTrainingCollator(collator_patch_size)
            else:
                naflex_collator = NaFlexMixupTrainingCollator(
                    patch_size=collator_patch_size,
                    num_classes=num_outputs,
                    alpha=args.mixup_alpha,
                    p=args.mixup_cutmix_prob,
                )

            if naflex_transforms is not None:
                naflex_batch_processor = NaFlexBatchProcessor(naflex_collator, naflex_specs, naflex_transforms, seed=0)
                if args.wds is False:
                    dataset = NaFlexMultiScaleDataset(dataset, naflex_batch_processor)

                collate_fn = naflex_batch_processor.base_collator
            else:
                collate_fn = naflex_collator
        else:
            t = get_mixup_cutmix(args.mixup_alpha, num_outputs, args.cutmix, prob=args.mixup_cutmix_prob)

            def mixup_cutmix_collate_fn(batch: Any) -> Any:
                return t(*default_collate(batch))

            collate_fn = mixup_cutmix_collate_fn

        if args.wds is True:
            wds_batcher: Optional[Callable[..., Iterator[tuple[Any, ...]]]] = None
            wds_num_shards: Optional[int] = None
            if naflex_batch_processor is not None:
                wds_batcher = partial(
                    naflex_batch_processor.iter_batches,
                    batch_size=batch_size,
                    drop_last=False,
                )
                wds_num_shards = get_wds_num_shards(dataset)

            data_loader = make_wds_loader(
                dataset,
                batch_size,
                num_workers=1,
                prefetch_factor=1,
                collate_fn=None if naflex_batch_processor is not None else collate_fn,
                world_size=1,
                pin_memory=False,
                shuffle=args.wds_extra_shuffle,
                batcher=wds_batcher,
                num_shards=wds_num_shards,
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
        mask_generator = _get_mask_generator(args, mask_size)

        for k, (inputs, _) in enumerate(data_loader):
            if k >= no_iterations:
                break

            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)
            batch_patch_size = args.patch_size

            if args.naflex is True:
                patches, grid_sizes, valid_mask = inputs
                batch_patch_size = math.isqrt(patches.size(2) // args.channels)
                if mask_generator is not None:
                    masks = mask_generator(patches.size(0), grid_sizes=grid_sizes)
                    patches = masking.mask_tokens(patches, masks)

                images = []
                for idx, grid_size in enumerate(grid_sizes):
                    image_patches = patches[idx]
                    image_valid_mask = valid_mask[idx]
                    images.append(
                        _unpatchify_naflex(image_patches, grid_size, image_valid_mask, channels=args.channels)
                    )

            elif mask_generator is not None:
                masks = mask_generator(inputs.size(0))
                inputs = masking.mask_tensor(inputs, masks, patch_factor=args.patch_size)
                images = inputs
            else:
                images = inputs

            # Show transformed
            counter = 0
            for i in range(cols):
                for j in range(rows):
                    img = images[i + cols * j]
                    transformed_img = F.to_pil_image(reverse_transform(img))

                    ax = fig.add_subplot(grid_spec[j, i])
                    ax.imshow(np.asarray(transformed_img))
                    if args.naflex is True:
                        ax.set_title(
                            f"#{counter}, {transformed_img.height}x{transformed_img.width}, p{batch_patch_size}"
                        )
                    else:
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
            "python -m birder.tools show-iterator --mode training --size 224 --patch-size 16 --naflex\n"
            "python -m birder.tools show-iterator --mode training --size 256 --patch-size 16 --naflex --batch "
            "--mixup-alpha 0.2\n"
            "python -m birder.tools show-iterator --mode training --size 256 --patch-size 16 --re-prob 0 --naflex "
            "--naflex-sizes 192 256 384 --naflex-patch-sizes 12 16 20 --batch --masking uniform "
            "--min-mask-size 2\n"
            "python -m birder.tools show-iterator --mode training --size 224 --aug-level 2 --batch\n"
            "python -m birder.tools show-iterator --mode inference --size 320\n"
            "python -m birder.tools show-iterator --mode training --size 224 --batch --wds "
            "--wds-class-file ~/Datasets/imagenet-1k-wds/classes.txt --wds-size 50000 "
            "--data-path ~/Datasets/imagenet-1k-wds/validation\n"
            "python -m birder.tools show-iterator --mode training --batch --size 224 --aug-level 1 --masking uniform\n"
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
    training_cli.add_data_aug_args(subparser)
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio during inference")
    subparser.add_argument(
        "--batch", default=False, action="store_true", help="show a batch instead of a single sample"
    )
    subparser.add_argument("--naflex", default=False, action="store_true", help="use native aspect-ratio transforms")
    subparser.add_argument(
        "--naflex-sizes",
        type=int,
        nargs="+",
        metavar="SIZE",
        help=(
            "square-equivalent image sizes in pixels to sample once per training batch, actual image "
            "dimensions preserve aspect ratio (values must be divisible by --patch-size)"
        ),
    )
    subparser.add_argument(
        "--naflex-patch-sizes",
        type=int,
        nargs="+",
        metavar="PATCH_SIZE",
        help="patch sizes to sample once per training batch (values do not need to divide the image sizes)",
    )
    subparser.add_argument("--mixup-alpha", type=float, help="mixup alpha")
    subparser.add_argument("--cutmix", default=False, action="store_true", help="enable cutmix")
    subparser.add_argument(
        "--mixup-cutmix-prob",
        type=float,
        metavar="P",
        help=(
            "probability of applying MixUp or CutMix to a batch "
            "(default: equal probability among enabled augmentations and no augmentation)"
        ),
    )
    subparser.add_argument(
        "--masking",
        type=str,
        choices=["uniform", "block", "fixed-size-block", "roll-block", "inverse-roll"],
        help="masking strategy to apply",
    )
    subparser.add_argument("--mask-ratio", type=float, default=0.5, help="mask ratio")
    subparser.add_argument(
        "--min-mask-size", type=int, default=1, help="minimum mask unit size in patches (uniform only)"
    )
    subparser.add_argument("--patch-size", type=int, default=16, help="patch size for masking and NaFlex")
    subparser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="image directory path"
    )
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
    subparser.add_argument("--wds", default=False, action="store_true", help="use webdataset")
    subparser.add_argument(
        "--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths"
    )
    subparser.add_argument("--wds-class-file", type=str, metavar="FILE", help="class list file")
    subparser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds directory")
    subparser.add_argument(
        "--wds-split", type=str, default="training", metavar="NAME", help="wds dataset split to load"
    )
    subparser.add_argument(
        "--wds-extra-shuffle",
        default=False,
        action="store_true",
        help="enable cross-worker batch shuffling after batching",
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.wds is True and args.batch is False:
        raise cli.ValidationError("--wds requires --batch to be set")
    if args.masking is not None and args.batch is False:
        raise cli.ValidationError("--masking requires --batch to be set")
    if args.naflex is True and args.cutmix is True:
        raise cli.ValidationError("--naflex cannot be used with --cutmix")
    if args.naflex is True and args.simple_crop is True:
        raise cli.ValidationError("--naflex does not support --simple-crop")
    if args.naflex is True and args.wds is True and args.wds_extra_shuffle is True:
        raise cli.ValidationError("--naflex cannot be used with --wds-extra-shuffle")
    if args.mixup_cutmix_prob is not None:
        if args.mixup_cutmix_prob < 0.0 or args.mixup_cutmix_prob > 1.0:
            raise cli.ValidationError(f"--mixup-cutmix-prob must be in range of [0, 1], got {args.mixup_cutmix_prob}")
        if args.mixup_alpha is None and args.cutmix is False:
            raise cli.ValidationError("--mixup-cutmix-prob requires --mixup-alpha or --cutmix")
    if args.naflex_sizes is not None and args.naflex is False:
        raise cli.ValidationError("--naflex-sizes requires --naflex")
    if args.naflex_sizes is not None and args.mode != "training":
        raise cli.ValidationError("--naflex-sizes is only supported in training mode")
    if args.naflex_patch_sizes is not None:
        if args.naflex is False:
            raise cli.ValidationError("--naflex-patch-sizes requires --naflex")
        if args.mode != "training":
            raise cli.ValidationError("--naflex-patch-sizes is only supported in training mode")
        if len(set(args.naflex_patch_sizes)) != len(args.naflex_patch_sizes):
            raise cli.ValidationError(f"--naflex-patch-sizes values must be unique, got {args.naflex_patch_sizes}")
    if args.rgb_mean is not None and len(args.rgb_mean) != args.channels:
        raise cli.ValidationError(f"--rgb-mean must have {args.channels} values, got {len(args.rgb_mean)}")
    if args.rgb_std is not None and len(args.rgb_std) != args.channels:
        raise cli.ValidationError(f"--rgb-std must have {args.channels} values, got {len(args.rgb_std)}")

    args.size = cli.parse_size(args.size)
    if args.naflex is True:
        if args.patch_size <= 0:
            raise cli.ValidationError("--patch-size must be positive")
        if args.size[0] % args.patch_size != 0 or args.size[1] % args.patch_size != 0:
            raise cli.ValidationError("--size must be divisible by --patch-size when using --naflex")

    show_iterator(args)
