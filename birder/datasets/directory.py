from collections.abc import Callable
from typing import Any
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from torchvision.io import decode_image

from birder.common import fs_ops


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


def make_image_dataset(
    paths: list[str],
    class_to_idx: dict[str, int],
    transforms: Optional[Callable[..., torch.Tensor]] = None,
    loader: Callable[[str], torch.Tensor] = decode_image,
) -> ImageListDataset:
    samples = fs_ops.samples_from_paths(paths, class_to_idx=class_to_idx)
    dataset = ImageListDataset(samples, transforms=transforms, loader=loader)

    return dataset
