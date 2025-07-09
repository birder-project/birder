import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import functional as F

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
