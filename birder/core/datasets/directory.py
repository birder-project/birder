from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.utils.data
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.v2.functional import pil_to_tensor


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        transforms: Optional[Callable[..., torch.Tensor]] = None,
        loader: Callable[[str], Any] = pil_loader,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.transforms = transforms
        self.loader = loader

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        (path, label) = self.samples[index]
        img = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(img)

        else:
            sample = pil_to_tensor(img)

        return (path, sample, label)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of data points: {self.__len__()}"]
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)
