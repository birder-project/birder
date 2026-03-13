import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from birder.conf import settings
from birder.datahub._lib import download_url
from birder.datahub._lib import extract_archive

SplitType = Literal["training", "validation", "testing"]


class TestDataset(ImageFolder):
    """
    Name: Birder TestDataset
    """

    def __init__(
        self,
        target_dir: Optional[str | Path] = None,
        download: bool = False,
        split: SplitType = "training",
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        progress_bar: bool = True,
    ) -> None:
        if target_dir is None:
            target_dir = settings.DATA_DIR

        if isinstance(target_dir, str):
            target_dir = Path(target_dir)

        self._target_dir: Path = target_dir
        self._root = self._target_dir.joinpath("TestDataset")
        if download is True:
            src = self._target_dir.joinpath("TestDataset.tar")
            downloaded = download_url(
                "https://f000.backblazeb2.com/file/birder/data/TestDataset.tar",
                src,
                sha256="28ca71c6308742ad8ebb47d5bd3de4db1f26b3173fa568d00abc1df05b556916",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(src, self._root)

        else:
            # Some sanity checks
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            for split_name in ["training", "validation"]:
                if self._root.joinpath(split_name).exists() is False:
                    raise RuntimeError("Dataset seems corrupted")

        super().__init__(self._root.joinpath(split), transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (path, sample, target)


class Flowers102(ImageFolder):
    """
    Name: 102 Flowers
    Link: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
    Size: 102 categories dataset consists of between 40 and 258 images per class (337MB)
    """

    def __init__(
        self,
        target_dir: Optional[str | Path] = None,
        download: bool = False,
        split: SplitType = "training",
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        progress_bar: bool = True,
    ) -> None:
        if target_dir is None:
            target_dir = settings.DATA_DIR

        if isinstance(target_dir, str):
            target_dir = Path(target_dir)

        self._target_dir: Path = target_dir
        self._root = self._target_dir.joinpath("Flowers102")
        if download is True:
            src = self._target_dir.joinpath("Flowers102.tar")
            downloaded = download_url(
                "https://f000.backblazeb2.com/file/birder/data/Flowers102.tar",
                src,
                sha256="6e7cf83821ed267e178dbabdb66cb4e23643aaf9d6180c3d93929f11cfbc4582",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(src, self._root)

        else:
            # Some sanity checks
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            for split_name in typing.get_args(SplitType):
                if self._root.joinpath(split_name).exists() is False:
                    raise RuntimeError("Dataset seems corrupted")

        super().__init__(self._root.joinpath(split), transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (path, sample, target)


# pylint: disable=invalid-name
class CUB_200_2011(ImageFolder):
    """
    Name: CUB_200_2011
    Link: https://www.vision.caltech.edu/datasets/cub_200_2011/
    Size: 200 categories dataset consists of 11,788 images (1.1GB)
    """

    def __init__(
        self,
        target_dir: Optional[str | Path] = None,
        download: bool = False,
        split: SplitType = "training",
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        progress_bar: bool = True,
    ) -> None:
        if target_dir is None:
            target_dir = settings.DATA_DIR

        if isinstance(target_dir, str):
            target_dir = Path(target_dir)

        self._target_dir: Path = target_dir
        self._root = self._target_dir.joinpath("CUB_200_2011")
        if download is True:
            src = self._target_dir.joinpath("CUB_200_2011.tar")
            downloaded = download_url(
                "https://huggingface.co/datasets/birder-project/CUB_200_2011/resolve/main/CUB_200_2011.tar",
                src,
                sha256="acb58211efa4253d59935572b6d1d3b9f6990c569d1cd318e2e1613d0a065916",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(src, self._root)

        else:
            # Some sanity checks
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            for split_name in ["training", "validation"]:
                if self._root.joinpath(split_name).exists() is False:
                    raise RuntimeError("Dataset seems corrupted")

        super().__init__(self._root.joinpath(split), transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (path, sample, target)


class ButterfliesMothsAustria(ImageFolder):
    """
    Name: butterflies-moths-austria
    Link: https://figshare.com/s/e79493adf7d26352f0c7
    Source: The original dataset contains 541,677 JPEG images covering 185 butterfly and moth species from Austria.
    Mirror: This re-encoded 57GB version is pre-split into training/validation/testing (70:20:10) and removes 6 classes
            with fewer than 10 images, leaving 179 classes.
            For more information, see https://huggingface.co/datasets/birder-project/butterflies-moths-austria
    """

    def __init__(
        self,
        target_dir: Optional[str | Path] = None,
        download: bool = False,
        split: SplitType = "training",
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        progress_bar: bool = True,
    ) -> None:
        if target_dir is None:
            target_dir = settings.DATA_DIR

        if isinstance(target_dir, str):
            target_dir = Path(target_dir)

        self._target_dir: Path = target_dir
        self._root = self._target_dir.joinpath("butterflies-moths-austria")
        if download is True:
            src = self._target_dir.joinpath("butterflies-moths-austria.tar")
            downloaded = download_url(
                (
                    "https://huggingface.co/datasets/birder-project/butterflies-moths-austria"
                    "/resolve/main/butterflies-moths-austria.tar"
                ),
                src,
                sha256="b3ad3f8b4e681328ee843158d8ff533c4cff0e4d777536161af15d3db8f7ff51",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(src, self._root)

        else:
            # Some sanity checks
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            for split_name in typing.get_args(SplitType):
                if self._root.joinpath(split_name).exists() is False:
                    raise RuntimeError("Dataset seems corrupted")

        super().__init__(self._root.joinpath(split), transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (path, sample, target)
