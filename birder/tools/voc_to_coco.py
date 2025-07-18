import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from tqdm import tqdm

import birder
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings

try:
    from defusedxml.ElementTree import parse

    _HAS_DEFUSEDXML = True
except ImportError:
    _HAS_DEFUSEDXML = False

logger = logging.getLogger(__name__)


def _create_annotation(
    points: tuple[float, float, float, float], label: int, image_id: int, annotation_id: int
) -> dict[str, Any]:
    annotation: dict[str, Any] = {}
    annotation["iscrowd"] = 0
    annotation["image_id"] = image_id

    # Bounding box in (x, y, w, h) format
    (xmin, ymin, xmax, ymax) = points
    assert xmax > xmin and ymax > ymin
    w = xmax - xmin
    h = ymax - ymin
    annotation["bbox"] = [xmin, ymin, w, h]
    annotation["category_id"] = label
    annotation["id"] = annotation_id

    return annotation


# pylint: disable=too-many-locals
def voc_to_coco(args: argparse.Namespace) -> None:
    prefix = str(settings.DETECTION_DATA_PATH) + "/"
    target_path = Path(args.ann_dir).parent.joinpath(f"{Path(args.class_file).stem}_coco.json")
    if target_path.exists() is True:
        logger.warning(f"{target_path} already exists, aborting...")

    class_to_idx = fs_ops.read_class_file(args.class_file)
    class_to_idx = lib.detection_class_to_idx(class_to_idx)

    img_path = Path(os.path.relpath(args.data_path, start=Path(args.ann_dir).parent))

    image_list = []
    annotation_list = []
    annotation_id = 0
    for idx, xml_path in tqdm(enumerate(fs_ops.file_iter(args.ann_dir, extensions=[".xml"])), leave=False):
        ann_tree = parse(xml_path)
        ann_root = ann_tree.getroot()

        size = ann_root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        filename = ann_root.findtext("filename")
        image_path = img_path.joinpath(filename)

        skip = False
        for obj in ann_root.findall("object"):
            label = obj.findtext("name")
            if label not in class_to_idx:
                logger.debug(f"Found unknown label: {label}, skipping image {img_path}")
                skip = True
                break

        if skip is True:
            continue

        image = {
            "id": idx,
            "width": width,
            "height": height,
            "file_name": os.path.normpath(image_path).removeprefix(prefix),
        }
        image_list.append(image)

        for obj in ann_root.findall("object"):
            label = obj.findtext("name")
            if label not in class_to_idx:
                logger.error(f"Found unknown label: {label}, aborting...")
                raise SystemExit(1)

            bbox = obj.find("bndbox")

            # VOC starts from (1, 1) instead of (0, 0)
            xmin = int(float(bbox.findtext("xmin"))) - 1
            ymin = int(float(bbox.findtext("ymin"))) - 1
            xmax = int(float(bbox.findtext("xmax")))
            ymax = int(float(bbox.findtext("ymax")))
            annotation_list.append(
                _create_annotation((xmin, ymin, xmax, ymax), class_to_idx[label], idx, annotation_id)
            )
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
        "birder_version": birder.__version__,
        "year": date.today().year,
        "date_created": date.today().isoformat(),
    }
    coco["images"] = image_list
    coco["categories"] = category_list
    coco["annotations"] = annotation_list

    # Save
    logger.info(f"Saving COCO file at {target_path}...")
    with open(target_path, "w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)

    logger.info(f"Written {len(image_list)} images with {len(annotation_list)} annotations")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "voc-to-coco",
        allow_abbrev=False,
        help="convert voc detection annotations to coco format",
        description="convert voc detection annotations to coco format",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools voc-to-coco --class-file public_datasets_metadata/voc-classes.txt "
            "--ann-dir ~/Datasets/VOC2012/Annotations ~/Datasets/VOC2012/JPEGImages\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--class-file", type=str, required=True, help="class list file")
    subparser.add_argument("--ann-dir", type=str, required=True, help="annotations directory")
    subparser.add_argument("data_path", help="image directory path")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert _HAS_DEFUSEDXML, "'pip install defusedxml' to use voc-to-coco"
    voc_to_coco(args)
