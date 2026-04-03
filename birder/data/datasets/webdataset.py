import logging
import os
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import torch
import webdataset as wds
from torchvision import tv_tensors
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.io import ImageReadMode
from torchvision.io import decode_image

from birder.common import fs_ops
from birder.common.training_utils import reduce_across_processes
from birder.conf import settings

logger = logging.getLogger(__name__)

WDS_SHUFFLE_SIZE = int(os.environ.get("WDS_SHUFFLE_SIZE", 4000))
WDS_INITIAL_SIZE = int(os.environ.get("WDS_INITIAL_SIZE", 1000))


def decode_sample_name(item: tuple[str, str, Any, int]) -> tuple[str, Any, int]:
    sample_name = item[0] + "/" + item[1]
    return (sample_name, item[2], item[3])


def decode_fake_cls(item: tuple[Any, ...]) -> tuple[Any, ...]:
    return (*item, settings.NO_LABEL)


def wds_image_decoder(key: str, data: bytes) -> torch.Tensor:
    if key.endswith(IMG_EXTENSIONS) is False:
        return None

    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return decode_image(tensor, mode=ImageReadMode.RGB)


def identity(x: Any) -> Any:
    return x


def make_wds_dataset(
    wds_path: str | list[str],
    dataset_size: int,
    shuffle: bool,
    samples_names: bool,
    transform: Callable[..., torch.Tensor],
    img_loader: Literal["tv", "pil"] = "tv",
    *,
    cls_key: Optional[str] = "cls",
    cache_dir: Optional[str] = None,
    shuffle_buffer_size: Optional[int] = None,
    shuffle_initial_size: Optional[int] = None,
) -> torch.utils.data.IterableDataset:
    if shuffle is True:
        shardshuffle = 500
    else:
        shardshuffle = False

    dataset = wds.WebDataset(
        wds_path, shardshuffle=shardshuffle, nodesplitter=wds.split_by_node, cache_dir=cache_dir, empty_check=False
    )
    if shuffle is True:
        if shuffle_buffer_size is None:
            shuffle_buffer_size = WDS_SHUFFLE_SIZE
        if shuffle_initial_size is None:
            shuffle_initial_size = WDS_INITIAL_SIZE

        logger.debug(f"Using buffer size of {shuffle_buffer_size} for shuffle with {shuffle_initial_size} initial size")
        dataset = dataset.shuffle(shuffle_buffer_size, initial=shuffle_initial_size)

    return_keys = ["jpeg;jpg;png;webp"]
    if cls_key is not None:
        return_keys = return_keys + [cls_key]
    if samples_names is True:
        return_keys = ["__url__", "__key__"] + return_keys

    if img_loader == "pil":
        dataset = dataset.with_length(dataset_size, silent=True).decode("pil").to_tuple(*return_keys)
    else:
        dataset = dataset.with_length(dataset_size, silent=True).decode(wds_image_decoder).to_tuple(*return_keys)

    if cls_key is None:
        dataset = dataset.map(decode_fake_cls)

    if samples_names is True:
        dataset = dataset.map(decode_sample_name)
        dataset = dataset.map_tuple(identity, transform, identity)
    else:
        dataset = dataset.map_tuple(transform, identity)

    return dataset


def decode_detection_target(item: tuple[Any, ...], label_remap: Optional[dict[int, int]] = None) -> tuple[Any, ...]:
    image = item[0]
    target = item[1]
    canvas_size = tuple(image.shape[-2:])

    if len(target["boxes"]) > 0:
        raw_labels = target["labels"]
        if label_remap is not None:
            raw_labels = [label_remap[label] for label in raw_labels]

        boxes = tv_tensors.BoundingBoxes(
            target["boxes"],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=canvas_size,
        )
        labels = torch.tensor(raw_labels, dtype=torch.int64)
    else:
        boxes = tv_tensors.BoundingBoxes(
            torch.zeros((0, 4), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=canvas_size,
        )
        labels = torch.zeros((0,), dtype=torch.int64)

    return (image, {"image_id": target["image_id"], "boxes": boxes, "labels": labels})


def decode_detection_sample(item: tuple[Any, ...], label_remap: Optional[dict[int, int]] = None) -> tuple[Any, ...]:
    if len(item) == 4:
        sample_name = item[0] + "/" + item[1]
        image, target = decode_detection_target((item[2], item[3]), label_remap=label_remap)
        orig_size = tuple(image.shape[-2:])
        return (sample_name, image, target, orig_size)

    image, target = decode_detection_target(item, label_remap=label_remap)
    orig_size = tuple(image.shape[-2:])
    return (image, target, orig_size)


def make_wds_detection_dataset(
    wds_path: str | list[str],
    dataset_size: int,
    shuffle: bool,
    transform: Callable[..., torch.Tensor],
    *,
    samples_names: bool = False,
    return_orig_sizes: bool = False,
    label_remap: Optional[dict[int, int]] = None,
    cache_dir: Optional[str] = None,
    shuffle_buffer_size: Optional[int] = None,
    shuffle_initial_size: Optional[int] = None,
) -> torch.utils.data.IterableDataset:
    if shuffle is True:
        shardshuffle = 500
    else:
        shardshuffle = False

    dataset = wds.WebDataset(
        wds_path, shardshuffle=shardshuffle, nodesplitter=wds.split_by_node, cache_dir=cache_dir, empty_check=False
    )
    if shuffle is True:
        if shuffle_buffer_size is None:
            shuffle_buffer_size = WDS_SHUFFLE_SIZE
        if shuffle_initial_size is None:
            shuffle_initial_size = WDS_INITIAL_SIZE

        logger.debug(f"Using buffer size of {shuffle_buffer_size} for shuffle with {shuffle_initial_size} initial size")
        dataset = dataset.shuffle(shuffle_buffer_size, initial=shuffle_initial_size)

    return_keys = ["jpeg;jpg;png;webp", "json"]
    if samples_names is True:
        return_keys = ["__url__", "__key__"] + return_keys

    dataset = dataset.with_length(dataset_size, silent=True).decode(wds_image_decoder).to_tuple(*return_keys)
    dataset = dataset.map(lambda item: decode_detection_sample(item, label_remap=label_remap))

    if samples_names is True:
        if return_orig_sizes is True:
            dataset = dataset.map(lambda item: (item[0], *transform(item[1], item[2]), item[3]))
        else:
            dataset = dataset.map(lambda item: (item[0], *transform(item[1], item[2])))
    else:
        if return_orig_sizes is True:
            dataset = dataset.map(lambda item: (*transform(item[0], item[1]), item[2]))
        else:
            dataset = dataset.map(lambda item: transform(item[0], item[1]))

    return dataset


def wds_size(wds_path: str, device: torch.device, select_suffix: str = "cls") -> int:
    dataset = wds.WebDataset(
        wds_path,
        shardshuffle=False,
        select_files=lambda key_name: key_name.endswith(select_suffix),
        nodesplitter=wds.split_by_node,
        empty_check=False,
    ).batched(64, collation_fn=None, partial=True)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=8)
    size = 0
    for batch in dataloader:
        size += len(batch)

    size = reduce_across_processes(size, device, op=torch.distributed.ReduceOp.SUM)  # type: ignore

    return size


def prepare_wds_args(
    data_path: str, size: Optional[int], device: torch.device, select_suffix: str = "cls"
) -> tuple[str, int]:
    if "://" not in data_path:
        if ".." not in data_path:
            # Local path without braces, build brace argument
            wds_path, _ = fs_ops.wds_braces_from_path(Path(data_path))

        else:
            wds_path = data_path

        if size is None:
            # If size not provided, we scan all tar files
            dataset_size = wds_size(wds_path, device, select_suffix=select_suffix)
        else:
            dataset_size = size

    else:
        # Remote path, we take the string as-is
        # Dataset size and must be provided
        assert size is not None
        wds_path = data_path
        dataset_size = size

    return (wds_path, dataset_size)


def _resolve_wds_info_filename(info_path: str | Path, filename: str) -> str:
    if "://" in filename:
        return filename

    info_path = str(info_path)
    if "://" in info_path:
        return info_path.rsplit("/", 1)[0] + "/" + filename

    return os.path.join(Path(info_path).parent, filename)


def wds_args_from_info(info_path: str | Path | Sequence[str | Path], split: str) -> tuple[list[str], int]:
    info_paths: list[str | Path]
    if isinstance(info_path, (str, Path)):
        info_paths = [info_path]
    else:
        info_paths = list(info_path)

    filenames: list[str] = []
    size = 0
    for current_info_path in info_paths:
        info = fs_ops.read_wds_info(current_info_path)
        split_info = info["splits"][split]

        size += split_info["num_samples"]
        filenames.extend(
            _resolve_wds_info_filename(current_info_path, filename) for filename in split_info["filenames"]
        )

    return (filenames, size)
