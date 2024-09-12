import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings


def _create_annotation(
    points: tuple[tuple[float, float], tuple[float, float]], label: int, image_id: int, annotation_id: int
) -> dict[str, Any]:
    annotation: dict[str, Any] = {}
    annotation["iscrowd"] = 0
    annotation["image_id"] = image_id

    # Bounding box in (x, y, w, h) format
    (x0, y0) = points[0]
    (x1, y1) = points[1]
    x = min(x0, x1)
    y = min(y0, y1)
    w = abs(x0 - x1)
    h = abs(y0 - y1)
    annotation["bbox"] = [x, y, w, h]
    annotation["category_id"] = label
    annotation["id"] = annotation_id

    return annotation


def labelme_to_coco(args: argparse.Namespace, target_path: Path) -> None:
    class_to_idx = fs_ops.read_class_file(settings.DETECTION_DATA_PATH.joinpath(settings.CLASS_LIST_NAME))
    class_to_idx = lib.detection_class_to_idx(class_to_idx)

    image_list = []
    annotation_list = []
    annotation_id = 0
    for idx, json_path in tqdm(enumerate(fs_ops.file_iter(args.data_path, extensions=[".json"])), leave=False):
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        unknown = data["flags"].get("unknown", False)
        if unknown is True and args.include_unknown is False:
            continue

        image = {
            "id": idx,
            "width": data["imageWidth"],
            "height": data["imageHeight"],
            "file_name": f"{args.data_path}/{data['imagePath']}",
        }
        image_list.append(image)

        for shapes in data["shapes"]:
            if shapes["shape_type"] != "rectangle":
                logging.error("Only detection rectangles are supported, aborting...")
                raise SystemExit(1)

            label = shapes["label"]
            if label not in class_to_idx:
                logging.error(f"Found unknown label: {label}, aborting...")
                raise SystemExit(1)

            points = shapes["points"]
            annotation_list.append(_create_annotation(points, class_to_idx[label], idx, annotation_id))
            annotation_id += 1

    # Create categories
    category_list = []
    for class_name, class_id in class_to_idx.items():
        category_list.append(
            {
                "supercategory": class_name,
                "id": class_id,
                "name": class_name,
            }
        )

    # Create COCO format dictionary
    coco: dict[str, Any] = {}
    coco["info"] = {
        "version": "1.0",
        "year": date.today().year,
        "date_created": date.today().isoformat(),
    }
    coco["images"] = image_list
    coco["categories"] = category_list
    coco["annotations"] = annotation_list

    # Save
    logging.info(f"Saving COCO file at {target_path}...")
    with open(target_path, "w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)

    logging.info(f"Written {len(image_list)} images with {len(annotation_list)} annotations")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "labelme-to-coco",
        allow_abbrev=False,
        help="convert labelme detection annotations to coco format",
        description="convert labelme detection annotations to coco format",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools labelme-to-coco data/detection_data/training_annotations\n"
            "python -m birder.tools labelme-to-coco --include-unknown data/detection_data/validation_annotations\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--include-unknown", default=False, action="store_true", help="include files with unknown flag"
    )
    subparser.add_argument("data_path", help="image directory path")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    base_name = Path(args.data_path).stem
    target_path = Path(args.data_path).parent.joinpath(f"{base_name}_coco.json")
    labelme_to_coco(args, target_path)
