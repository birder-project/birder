import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from birder.core.datahub._lib import download_url
from birder.core.datahub._lib import extract_archive

SplitType = Literal["training", "validation", "testing"]


class Caltech256:
    """
    Name: Caltech 256
    Link: https://data.caltech.edu/records/20087
    License: Creative Commons Attribution 4.0 International
    Size: 256 object categories containing a total of 30607 images (1.2GB)
    """

    def __init__(self) -> None:
        raise NotImplementedError


class Flowers102(ImageFolder):
    """
    Name 102 Flowers
    Link: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
    Size: 102 category dataset consists of between 40 and 258 images per class (337MB)
    """

    def __init__(
        self,
        target_dir: str | Path,
        download: bool = False,
        split: SplitType = "training",
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if isinstance(target_dir, str) is True:
            target_dir = Path(target_dir)

        self._target_dir: Path = target_dir  # type: ignore[assignment]
        self._root = self._target_dir.joinpath("Flowers102")
        if download is True:
            src = self._target_dir.joinpath("Flowers102.tar")
            download_url(
                "https://f000.backblazeb2.com/file/birder/data/Flowers102.tar",
                src,
                sha256="a585173c9ae604f3129d00a9aafc4d1851351e2590c20e3cbfe87f6d4ee41fb2",
            )
            extract_archive(src, self._target_dir)

        else:
            # Some sanity checks
            if self._target_dir.exists() is False or self._target_dir.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            for split_name in typing.get_args(SplitType):
                if self._root.joinpath(split_name).exists() is False:
                    raise RuntimeError("Dataset seems corrupted")

        super().__init__(self._root.joinpath(split), transform, target_transform, loader, is_valid_file)
