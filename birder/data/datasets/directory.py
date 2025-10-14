import os
from collections.abc import Callable
from typing import Any
from typing import Optional
from typing import cast

import numpy as np
import torch
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.io import ImageReadMode
from torchvision.io import decode_image
from torchvision.transforms.v2 import functional as F

from birder.common import fs_ops


def tv_loader(path: str) -> torch.Tensor:
    return decode_image(path, mode=ImageReadMode.RGB)


def default_is_valid_file(x: str) -> bool:
    return has_file_allowed_extension(x, IMG_EXTENSIONS)  # type: ignore[no-any-return]


def find_hierarchical_classes(
    directory: str, separator: str, is_valid_file: Optional[Callable[[str], bool]]
) -> tuple[list[str], dict[str, int]]:
    label_set = set()
    for root, _, files in os.walk(directory):
        rel_path = os.path.relpath(root, directory)
        if rel_path == ".":
            continue

        # Check if this directory contains any valid files
        if is_valid_file is not None:
            has_valid_files = any(is_valid_file(os.path.join(root, f)) for f in files)
        else:
            has_valid_files = True

        if has_valid_files is True:
            class_name = rel_path.replace(os.sep, separator)
            label_set.add(class_name)

    classes = sorted(list(label_set))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return (classes, class_to_idx)


class HierarchicalImageFolder(DatasetFolder):
    """
    A dataset class that supports hierarchical directory structures.

    Unlike the standard ImageFolder which uses only the immediate parent directory
    as the class label, this class creates labels from the full path hierarchy
    relative to the root directory.

    For example, given structure:
    root/
        dir1/
            subdir1/img1.jpeg -> label: "dir1_subdir1"
            subdir2/img2.jpeg -> label: "dir1_subdir2"
        dir2/
            subdir3/
                sub-subdir4/img3.jpeg -> label: "dir2_subdir3_sub-subdir4"
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[tuple[str, ...]] = None,
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., torch.Tensor]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        separator: str = "_",
    ):
        self.separator = separator
        if extensions is None:
            extensions = IMG_EXTENSIONS

        self.extensions = extensions
        super().__init__(
            root=root,
            loader=loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

    def _is_valid_file(
        self,
        file_path: str,
        extensions: Optional[tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        if is_valid_file is not None:
            return is_valid_file(file_path)

        if extensions is None:
            extensions = self.extensions

        return file_path.lower().endswith(extensions)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        return find_hierarchical_classes(directory, separator=self.separator, is_valid_file=self._is_valid_file)

    def make_dataset(  # pylint: disable=arguments-renamed
        self,
        directory: str,
        class_to_idx: dict[str, int],
        extensions: Optional[tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> list[tuple[str, int]]:
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none is True or both_something is True:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:  # pylint: disable=function-redefined
                return has_file_allowed_extension(x, extensions)  # type: ignore[no-any-return]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for root, _, file_names in sorted(os.walk(directory, followlinks=True)):
            rel_path = os.path.relpath(root, directory)
            if rel_path == ".":
                continue

            class_name = rel_path.replace(os.sep, self.separator)
            if class_name not in class_to_idx:
                continue

            class_index = class_to_idx[class_name]

            # Add all valid files in this directory
            for fname in sorted(file_names):
                file_path = os.path.join(root, fname)
                if self._is_valid_file(file_path, extensions, is_valid_file):
                    instances.append((file_path, class_index))
                    if class_name not in available_classes:
                        available_classes.add(class_name)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"

            raise FileNotFoundError(msg)

        return instances


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
    *,
    return_orig_sizes: bool = False,
    hierarchical: bool = False,
) -> ImageListDataset:
    samples = fs_ops.collect_samples_from_paths(paths, class_to_idx=class_to_idx, hierarchical=hierarchical)
    if return_orig_sizes is True:
        dataset = ImageListDatasetWithSize(samples, transforms=transforms, loader=loader)
    else:
        dataset = ImageListDataset(samples, transforms=transforms, loader=loader)

    return dataset


def class_to_idx_from_paths(data_paths: list[str], hierarchical: bool = False) -> dict[str, int]:
    class_to_idx = {}
    base = 0
    for data_path in data_paths:
        if hierarchical is True:
            classes = find_hierarchical_classes(data_path, separator="_", is_valid_file=default_is_valid_file)[0]
        else:
            classes = sorted(entry.name for entry in os.scandir(data_path) if entry.is_dir())

        class_to_idx.update({cls_name: i + base for i, cls_name in enumerate(classes)})
        base += len(classes)

    return class_to_idx
