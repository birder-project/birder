import argparse
import random
from pathlib import Path
from typing import Any
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.data.collators.detection import collate_fn
from birder.data.datasets.coco import CocoInference
from birder.data.datasets.coco import CocoTraining
from birder.data.datasets.directory import tv_loader
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import reverse_preset
from birder.data.transforms.detection import AugType
from birder.data.transforms.detection import InferenceTransform
from birder.data.transforms.detection import training_preset


# pylint: disable=too-many-locals,too-many-branches
def show_det_iterator(args: argparse.Namespace) -> None:
    reverse_transform = reverse_preset(get_rgb_stats("birder"))
    root_path = Path(args.data_path)
    if args.mode == "training":
        offset = 0
        transform = training_preset(
            args.size,
            args.aug_type,
            args.aug_level,
            get_rgb_stats("birder"),
            args.dynamic_size,
            args.multiscale,
            args.max_size,
        )
        dataset = CocoTraining(args.data_path, args.coco_json_path, transforms=transform)
    elif args.mode == "inference":
        offset = 1
        transform = InferenceTransform(args.size, get_rgb_stats("birder"), args.dynamic_size, args.max_size)
        dataset = CocoInference(args.data_path, args.coco_json_path, transforms=transform)
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    if args.class_file is not None:
        class_to_idx = fs_ops.read_class_file(args.class_file)
        class_to_idx = lib.detection_class_to_idx(class_to_idx)
    else:
        class_to_idx = lib.class_to_idx_from_coco(dataset.dataset.coco.cats)

    class_list = list(class_to_idx.keys())
    class_list.insert(0, "Background")
    color_list = np.arange(0, len(class_list))

    batch_size = 2
    no_iterations = 6
    if args.batch is False:
        samples = random.sample(dataset.dataset.ids, no_iterations)
        cols = 2
        rows = 2
        for coco_id in samples:
            img_path = dataset.dataset.coco.loadImgs(coco_id)[0]["file_name"]
            img = tv_loader(str(root_path.joinpath(img_path)))
            targets = dataset.dataset.coco.loadAnns(dataset.dataset.coco.getAnnIds(coco_id))

            # Roughly what the "wrap_dataset_for_transforms_v2" does
            canvas_size = tuple(F.get_size(img))
            boxes = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    [target["bbox"] for target in targets],
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            )

            label_ids = [target["category_id"] for target in targets]
            labels = [class_list[label_id] for label_id in label_ids]
            colors = [color_list[label_id].item() for label_id in label_ids]

            annotated_img = draw_bounding_boxes(img, boxes, labels=labels, colors=colors)
            transformed_img = F.to_pil_image(annotated_img)

            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

            # Show original
            ax = fig.add_subplot(grid_spec[0, 0:cols])
            ax.imshow(transformed_img)
            ax.set_title(f"Original, aug type: {args.aug_type}")

            # Show transformed
            counter = 0
            for i in range(cols):
                for j in range(1, rows):
                    idx = dataset.dataset.ids.index(coco_id)
                    data = dataset[idx]
                    img = data[offset]
                    targets = data[offset + 1]
                    label_ids = targets["labels"]
                    labels = [class_list[label_id] for label_id in label_ids]
                    colors = [color_list[label_id].item() for label_id in label_ids]

                    annotated_img = draw_bounding_boxes(
                        reverse_transform(img), targets["boxes"], labels=labels, colors=colors
                    )
                    transformed_img = F.to_pil_image(annotated_img)

                    ax = fig.add_subplot(grid_spec[j, i])
                    ax.imshow(np.asarray(transformed_img))
                    ax.set_title(f"#{counter}")
                    counter += 1

            plt.show()

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        cols = 2
        rows = 1
        for k, data in enumerate(data_loader):
            if k >= no_iterations:
                break

            inputs = data[offset]
            targets = data[offset + 1]
            if offset > 0:
                sample_paths = data[offset - 1]
            else:
                sample_paths = list(range(cols * rows))

            fig = plt.figure(constrained_layout=True)
            grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

            # Show transformed
            counter = 0
            for i in range(cols):
                for j in range(rows):
                    img = inputs[i + cols * j]
                    img = reverse_transform(img)
                    boxes = targets[i + cols * j]["boxes"]
                    label_ids = targets[i + cols * j]["labels"]
                    labels = [class_list[label_id] for label_id in label_ids]
                    colors = [color_list[label_id].item() for label_id in label_ids]

                    annotated_img = draw_bounding_boxes(img, boxes, labels=labels, colors=colors)
                    transformed_img = F.to_pil_image(annotated_img)
                    ax = fig.add_subplot(grid_spec[j, i])
                    ax.imshow(np.asarray(transformed_img))
                    ax.set_title(f"#{counter}: {sample_paths[i+cols*j]}")
                    counter += 1

            plt.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "show-det-iterator",
        allow_abbrev=False,
        help="show training / inference detection iterator output vs input",
        description="show training / inference detection iterator output vs input",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools show-det-iterator --aug-level 0\n"
            "python -m birder.tools show-det-iterator --mode training --size 512 --aug-level 2\n"
            "python -m birder.tools show-det-iterator --mode inference --size 640\n"
            "python -m birder.tools show-det-iterator --mode inference --coco-json-path "
            "~/Datasets/Objects365-2020/val/zhiyuan_objv2_val.json --data-path ~/Datasets/Objects365-2020/val\n"
            "python -m birder.tools show-det-iterator --aug-type ssd --dynamic-size --coco-json-path "
            "~/Datasets/cocodataset/annotations/instances_val2017.json --data-path "
            "~/Datasets/cocodataset/val2017 --class-file public_datasets_metadata/coco-classes.txt\n"
            "python tool.py show-det-iterator --mode training --aug-level 4 --multiscale "
            "--coco-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json "
            "--data-path ~/Datasets/cocodataset/val2017 --class-file public_datasets_metadata/coco-classes.txt\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--mode", type=str, choices=["training", "inference"], default="training", help="iterator mode"
    )
    subparser.add_argument(
        "--batch", default=False, action="store_true", help="show a batch instead of a single sample"
    )
    subparser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[512],
        metavar=("H", "W"),
        help=(
            "target image size as [height, width], if --dynamic-size is enabled, "
            "uses the smaller dimension as target size while preserving aspect ratio (defaults to model's signature)"
        ),
    )
    subparser.add_argument(
        "--max-size",
        type=int,
        help="maximum size for the longer edge of resized images, when specified, enables dynamic sizing",
    )
    subparser.add_argument(
        "--dynamic-size",
        default=False,
        action="store_true",
        help="allow variable image sizes while preserving aspect ratios",
    )
    subparser.add_argument("--multiscale", default=False, action="store_true", help="enable random scale per image")
    subparser.add_argument(
        "--aug-type",
        type=str,
        choices=list(get_args(AugType)),
        default="birder",
        help="augmentation type",
    )
    subparser.add_argument(
        "--aug-level",
        type=int,
        choices=list(range(10 + 1)),
        default=4,
        help="magnitude of birder augmentations (0 off -> 10 highest)",
    )
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
    subparser.add_argument("--class-file", type=str, metavar="FILE", help="class list file, overrides json categories")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.aug_type == "birder" or args.multiscale is False, "multiscale only supported for birder augmentations"
    args.size = cli.parse_size(args.size)
    show_det_iterator(args)
