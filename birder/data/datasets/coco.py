from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2


class CocoInference(torch.utils.data.Dataset):
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

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        coco_id = self.dataset.ids[index]
        path = self.dataset.coco.loadImgs(coco_id)[0]["file_name"]
        (sample, labels) = self.dataset[index]

        return (path, sample, labels)

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return repr(self.dataset)
