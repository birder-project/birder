import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import functional as F

from birder.data.transforms.mosaic import mosaic_fixed_grid
from birder.data.transforms.mosaic import mosaic_random_center

MosaicType = Literal["fixed_grid", "random_center"]

logger = logging.getLogger(__name__)


def _remove_images_without_annotations(dataset: CocoDetection, ignore_list: list[str]) -> CocoDetection:
    def _has_only_empty_bbox(anno: list[dict[str, Any]]) -> bool:
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno: list[dict[str, Any]]) -> bool:
        # If it's empty, there is no annotation
        if len(anno) == 0:
            return False

        # If all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        img_path = dataset.coco.loadImgs(img_id)[0]["file_name"]
        if img_path in ignore_list:
            logger.debug(f"Ignoring {img_path}")
            continue

        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    return torch.utils.data.Subset(dataset, ids)


def _convert_to_binary_annotations(dataset: CocoDetection) -> CocoDetection:
    for img_id in dataset.ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        for obj in anno:
            obj["category_id"] = 1

    return dataset


class CocoBase(torch.utils.data.Dataset):
    def __init__(
        self, root: str | Path, ann_file: str, transforms: Optional[Callable[..., torch.Tensor]] = None
    ) -> None:
        super().__init__()
        dataset = CocoDetection(root, ann_file, transforms=transforms)
        self.class_to_idx = {cat["name"]: cat["id"] for cat in dataset.coco.cats.values()}

        # The transforms v2 wrapper causes open files count to "leak"
        # It seems due to the Pythonic COCO objects, maybe related to
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        self.dataset = wrap_dataset_for_transforms_v2(dataset)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return repr(self.dataset)

    def remove_images_without_annotations(self, ignore_list: list[str]) -> None:
        self.dataset = _remove_images_without_annotations(self.dataset, ignore_list)

    def convert_to_binary_annotations(self) -> None:
        self.dataset = _convert_to_binary_annotations(self.dataset)
        self.class_to_idx = {"Object": 1}


class CocoTraining(CocoBase):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        (sample, labels) = self.dataset[index]
        return (sample, labels)


class CocoInference(CocoBase):
    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any, list[int]]:
        coco_id = self.dataset.ids[index]
        path = self.dataset.coco.loadImgs(coco_id)[0]["file_name"]
        (sample, labels) = self.dataset[index]

        return (path, sample, labels, F.get_size(sample))


class CocoMosaicTraining(CocoBase):
    def __init__(
        self,
        root: str | Path,
        ann_file: str,
        transforms: Optional[Callable[..., torch.Tensor]],
        mosaic_transforms: Optional[Callable[..., torch.Tensor]],
        output_size: tuple[int, int],
        fill_value: int | tuple[int, int, int],
        mosaic_prob: float,
        mosaic_type: MosaicType = "fixed_grid",
    ) -> None:
        super().__init__(root, ann_file, transforms=None)
        self.transforms = transforms
        self.mosaic_transforms = mosaic_transforms
        self.output_size = output_size
        self.fill_value = fill_value
        self.mosaic_prob = mosaic_prob
        self.mosaic_fn = mosaic_fixed_grid if mosaic_type == "fixed_grid" else mosaic_random_center

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        if torch.rand(1) < self.mosaic_prob:
            indices = [index] + torch.randint(len(self), (3,)).tolist()
            images = []
            targets = []
            for idx in indices:
                (img, target) = self.dataset[idx]
                images.append(img)
                targets.append(target)

            (img, target) = self.mosaic_fn(images, targets, self.output_size, self.fill_value)
            if self.mosaic_transforms is not None:
                (img, target) = self.mosaic_transforms(img, target)

        else:
            (img, target) = self.dataset[index]

            # When an image has no annotations, wrapped COCO returns only {'image_id': ...}
            if "boxes" not in target or len(target["boxes"]) == 0:
                canvas_size = F.get_size(img)
                boxes = tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=canvas_size,
                )
                labels = torch.zeros((0,), dtype=torch.int64)
                target = {"boxes": boxes, "labels": labels}

            if self.transforms is not None:
                (img, target) = self.transforms(img, target)

        return (img, target)
