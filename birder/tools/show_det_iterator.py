import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.utils import draw_bounding_boxes

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.transforms.classification import get_rgb_stats
from birder.transforms.classification import reverse_preset
from birder.transforms.detection import inference_preset
from birder.transforms.detection import training_preset


# pylint: disable=too-many-locals
def show_det_iterator(args: argparse.Namespace) -> None:
    reverse_transform = reverse_preset(get_rgb_stats("birder"))
    if args.mode == "training":
        transform = training_preset(args.size, args.aug_level, get_rgb_stats("birder"))
    elif args.mode == "inference":
        transform = inference_preset(args.size, get_rgb_stats("birder"))
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    batch_size = 2

    class_to_idx = fs_ops.read_class_file(settings.DETECTION_DATA_PATH.joinpath(settings.CLASS_LIST_NAME))
    class_to_idx = lib.detection_class_to_idx(class_to_idx)
    class_list = list(class_to_idx.keys())
    class_list.insert(0, "Background")
    color_list = np.arange(0, len(class_list))

    dataset = CocoDetection(args.data_path, args.coco_json_path, transforms=transform)
    dataset = wrap_dataset_for_transforms_v2(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    no_iterations = 6
    cols = 2
    rows = 1
    for k, (inputs, targets) in enumerate(data_loader):
        if k >= no_iterations:
            break

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
                ax.set_title(f"#{counter}")
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
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--mode", type=str, choices=["training", "inference"], default="training", help="iterator mode"
    )
    subparser.add_argument("--size", type=int, nargs="+", default=[512], metavar=("H", "W"), help="image size")
    subparser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="magnitude of augmentations (0 off -> 3 highest)",
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
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    show_det_iterator(args)
