from collections.abc import Callable
from typing import Any
from typing import Optional

import numpy as np
import torch
from torchvision.io import ImageReadMode
from torchvision.io import decode_image
from torchvision.transforms.v2 import functional as F

from birder.common import fs_ops


def tv_loader(path: str) -> torch.Tensor:
    return decode_image(path, mode=ImageReadMode.RGB)


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        transforms: Optional[Callable[..., torch.Tensor]] = None,
        loader: Callable[[str], torch.Tensor] = decode_image,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.loader = loader

        # Avoid yielding Python objects
        # see: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        (paths, labels) = list(zip(*samples))
        self.labels = np.array(labels, dtype=np.int32)
        self.paths = np.array(paths, dtype=np.bytes_)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        path = self.paths[index].decode("utf-8")
        label = self.labels[index].item()
        img = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(img)

        else:
            sample = img

        return (path, sample, label)

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of data points: {self.__len__()}"]
        if hasattr(self, "transforms") is True and self.transforms is not None:
            body += [repr(self.transforms)]

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)


class ImageListDatasetWithSize(ImageListDataset):
    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any, list[int]]:  # type: ignore[override]
        path = self.paths[index].decode("utf-8")
        label = self.labels[index].item()
        img = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(img)

        else:
            sample = img

        return (path, sample, label, F.get_size(img))


def make_image_dataset(
    paths: list[str],
    class_to_idx: dict[str, int],
    transforms: Optional[Callable[..., torch.Tensor]] = None,
    loader: Callable[[str], torch.Tensor] = decode_image,
    return_orig_sizes: bool = False,
) -> ImageListDataset:
    samples = fs_ops.samples_from_paths(paths, class_to_idx=class_to_idx)
    if return_orig_sizes is True:
        dataset = ImageListDatasetWithSize(samples, transforms=transforms, loader=loader)
    else:
        dataset = ImageListDataset(samples, transforms=transforms, loader=loader)

    return dataset
