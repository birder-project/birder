import argparse
import logging
import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from birder.common import cli
from birder.conf import settings
from birder.data.collators.detection import collate_fn

logger = logging.getLogger(__name__)


def verify_coco(args: argparse.Namespace) -> None:
    batch_size = 32

    transform = v2.Compose(
        [
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = CocoDetection(args.data_path, args.coco_json_path, transforms=transform)
    dataset = wrap_dataset_for_transforms_v2(dataset)
    total = len(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        drop_last=False,
    )

    with tqdm(total=total, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        if args.fast is True:
            idx = 0
            try:
                for idx, (_, _) in enumerate(data_loader):
                    progress.update(batch_size)

            except (OSError, RuntimeError) as e:
                logger.warning(
                    f"File at batch no. {idx} (batch size = {batch_size}, "
                    f"{(idx-1) * batch_size}) failed to load {e}"
                )

        else:
            for img_id in dataset.coco.imgs:
                try:
                    img_path = dataset.coco.loadImgs(img_id)[0]["file_name"]
                    img_path = os.path.join(dataset.root, img_path)
                    img = decode_image(img_path)
                    img = transform(img)
                    if img.size(0) != 3:
                        logger.warning(f"File {img_path} failed to load {img.size()}")

                except (OSError, RuntimeError) as e:
                    logger.warning(f"File {img_path} failed to load {e}")

                progress.update(1)

    logger.info("Finished")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "verify-coco",
        allow_abbrev=False,
        help="loads every image in the dataset and raises exception on corrupted or missing files",
        description="loads every image in the dataset and raises exception on corrupted or missing files",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools verify-coco\n"
            "python -m birder.tools verify-coco --coco-json-path "
            "data/detection_data/training_annotations_coco.json --data-path data/detection_data\n"
            "python -m birder.tools verify-coco --fast --coco-json-path "
            "~/Datasets/Objects365-2020/train/zhiyuan_objv2_train.json --data-path ~/Datasets/Objects365-2020/train\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--fast", default=False, action="store_true", help="use parallel dataloader")
    subparser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.DETECTION_DATA_PATH),
        help="image directory path",
    )
    subparser.add_argument(
        "--coco-json-path",
        type=str,
        default=f"{settings.TRAINING_DETECTION_ANNOTATIONS_PATH}_coco.json",
        help="training COCO json path",
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    verify_coco(args)
