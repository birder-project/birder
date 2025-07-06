import argparse
import json
import logging
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.conf import settings

logger = logging.getLogger(__name__)

# Few datasets like Objects365 have some very big files
Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 1024 // 4 // 3)
MAX_SIZE = 16_000


class CustomImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        *,
        class_to_idx: dict[str, int],
    ) -> None:
        self._class_to_idx = class_to_idx
        super().__init__(root, loader=str, allow_empty=True)

    def find_classes(self, _directory: str) -> tuple[list[str], dict[str, int]]:
        classes = list(self._class_to_idx.keys())
        return (classes, self._class_to_idx)


def _get_class_to_idx(paths: list[str]) -> dict[str, int]:
    class_list: list[str] = []
    for path in paths:
        dataset = ImageFolder(path)
        class_list.extend(list(dataset.class_to_idx.keys()))

    class_list = sorted(list(set(class_list)))
    class_to_idx = {k: v for v, k in enumerate(class_list)}

    return class_to_idx


def _save_classes(pack_path: Path, class_to_idx: dict[str, int]) -> None:
    class_list_path = pack_path.joinpath("classes.txt")
    doc = "\n".join(list(class_to_idx.keys()))

    logger.info(f"Saving class list at {class_list_path}")
    with open(class_list_path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def _encode_image(path: str, file_format: str, size: Optional[int] = None) -> bytes:
    image: Image.Image = Image.open(path)
    if size is not None and size < min(image.size):
        if image.size[0] > image.size[1]:
            ratio = image.size[1] / size
        else:
            ratio = image.size[0] / size

        width = round(image.size[0] / ratio)
        height = round(image.size[1] / ratio)
        if max(width, height) > MAX_SIZE:
            if width > height:
                ratio = width / MAX_SIZE
            else:
                ratio = height / MAX_SIZE

            width = round(width / ratio)
            height = round(height / ratio)

        image = image.resize((width, height), Image.Resampling.BICUBIC)

    elif max(image.size) > MAX_SIZE:
        if image.size[0] > image.size[1]:
            ratio = image.size[0] / MAX_SIZE
        else:
            ratio = image.size[1] / MAX_SIZE

        width = round(image.size[0] / ratio)
        height = round(image.size[1] / ratio)
        image = image.resize((width, height), Image.Resampling.BICUBIC)

    sample_buffer = BytesIO()
    image.save(sample_buffer, format=file_format, quality=85)
    return sample_buffer.getvalue()


def read_worker(q_in: Any, q_out: Any, error_event: Any, size: Optional[int], file_format: str) -> None:
    while True:
        deq: Optional[tuple[int, str, int]] = q_in.get()
        if deq is None:
            break

        try:
            (idx, path, target) = deq
            if size is None:
                suffix = Path(path).suffix[1:]
                if file_format != suffix:
                    sample = _encode_image(path, file_format)
                else:
                    with open(path, "rb") as stream:
                        sample = stream.read()

            else:
                sample = _encode_image(path, file_format, size)

        except Exception:
            error_event.set()
            raise

        if error_event.is_set() is True:
            break

        q_out.put((idx, sample, file_format, target), block=True, timeout=None)


def wds_write_worker(
    q_out: Any, error_event: Any, pack_path: Path, total: int, args: argparse.Namespace, _: dict[int, str]
) -> None:
    try:
        info_path = pack_path.joinpath("_info.json")
        if args.append is True:
            info = fs_ops.read_wds_info(info_path)
            if args.split in info["splits"]:
                raise ValueError(f"split {args.split} already exist")

        else:
            info = None
    except Exception:
        error_event.set()
        # Re-raise to terminate this process
        raise

    filenames: list[str] = []
    shard_lengths: list[int] = []
    path_pattern = str(pack_path.joinpath(f"{args.suffix}-{args.split}-%06d.tar"))
    sink = wds.ShardWriter(path_pattern, maxsize=args.max_size, verbose=0)

    def wds_info(fname: str) -> None:
        filenames.append(Path(fname).name)
        shard_lengths.append(sink.count)

    sink.post = wds_info

    count = 0
    buf = {}
    more = True
    with tqdm(total=total, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        while more:
            deq: Optional[tuple[int, bytes, str, int]] = q_out.get()
            if deq is not None:
                (idx, sample, suffix, target) = deq
                buf[idx] = (sample, suffix, target)

            else:
                more = False

            # Ensures ordered write
            while count in buf:
                (sample, suffix, target) = buf[count]
                del buf[count]
                sink.write(
                    {
                        "__key__": f"sample{count:08d}",
                        suffix: sample,
                        "cls": target,
                    }
                )

                count += 1

                # Update progress bar
                progress.update(n=1)

    sink.close()

    split_info = {
        "name": args.split,
        "filenames": filenames,
        "shard_lengths": shard_lengths,
        "num_samples": sum(shard_lengths),
    }

    if info is None:
        info = {
            "name": args.suffix,
            "splits": {args.split: split_info},
        }
    else:
        info["splits"][args.split] = split_info

    with open(pack_path.joinpath("_info.json"), "w", encoding="utf-8") as handle:
        logger.debug("Saving _info.json")
        json.dump(info, handle, indent=2)


def directory_write_worker(
    q_out: Any, _error_event: Any, pack_path: Path, total: int, _: argparse.Namespace, idx_to_class: dict[int, str]
) -> None:
    count = 0
    buf = {}
    more = True
    with tqdm(total=total, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        while more:
            deq: Optional[tuple[int, bytes, str, int]] = q_out.get()
            if deq is not None:
                (idx, sample, suffix, target) = deq
                buf[idx] = (sample, suffix, target)

            else:
                more = False

            # Ensures ordered write
            while count in buf:
                (sample, suffix, target) = buf[count]
                del buf[count]
                with open(pack_path.joinpath(idx_to_class[target]).joinpath(f"{count:06d}.{suffix}"), "wb") as handle:
                    handle.write(sample)

                count += 1

                # Update progress bar
                progress.update(n=1)


# pylint: disable=too-many-locals,too-many-branches
def pack(args: argparse.Namespace, pack_path: Path) -> None:
    if args.sampling_file is not None:
        with open(args.sampling_file, "r", encoding="utf-8") as handle:
            sampling_lines = handle.readlines()

        data_paths = []
        for line in sampling_lines:
            if len(line.strip()) == 0 or line.strip().startswith("#") is True:
                continue

            (data_path, r) = line.split()
            data_path = os.path.expanduser(data_path)
            repeats = int(r)
            for _ in range(repeats):
                data_paths.append(data_path)

    else:
        data_paths = args.data_path

    if args.no_cls is False:
        if args.class_file is not None:
            class_to_idx = fs_ops.read_class_file(args.class_file)
        elif args.append is True:
            class_to_idx = fs_ops.read_class_file(pack_path.joinpath("classes.txt"))
        else:
            class_to_idx = _get_class_to_idx(data_paths)

        _save_classes(pack_path, class_to_idx)
        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

        datasets = []
        for path in data_paths:
            datasets.append(CustomImageFolder(path, class_to_idx=class_to_idx))

        dataset = ConcatDataset(datasets)

    else:
        idx_to_class = {}
        dataset = fs_ops.samples_from_paths(data_paths, class_to_idx={})

    if args.shuffle is True:
        indices = torch.randperm(len(dataset)).tolist()
    else:
        indices = list(range(len(dataset)))

    if args.jobs == -1:
        args.jobs = multiprocessing.cpu_count()

    logger.info(f"Packing {len(dataset):,} samples")
    logger.info(f"Running {args.jobs} read processes and 1 write process")

    q_in = []  # type: ignore
    for _ in range(args.jobs):
        q_in.append(multiprocessing.Queue(1024))

    q_out = multiprocessing.Queue(1024)  # type: ignore
    error_event = multiprocessing.Event()

    read_processes: list[multiprocessing.Process] = []
    for idx in range(args.jobs):
        read_processes.append(
            multiprocessing.Process(
                target=read_worker, args=(q_in[idx], q_out, error_event, args.size, args.format), daemon=True
            )
        )

    for p in read_processes:
        p.start()

    if args.type == "wds":
        target_writer: Callable[..., None] = wds_write_worker
    elif args.type == "directory":
        target_writer = directory_write_worker
        for c in class_to_idx.keys():
            pack_path.joinpath(c).mkdir()
    else:
        raise ValueError("Unknown pack type")

    write_process = multiprocessing.Process(
        target=target_writer, args=(q_out, error_event, pack_path, len(dataset), args, idx_to_class), daemon=True
    )
    write_process.start()

    def signal_handler(signum, _frame) -> None:  # type: ignore
        logger.info(f"Received signal: {signum} at {multiprocessing.current_process().name}, aborting...")
        error_event.set()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, signal_handler)

    tic = time.time()
    for idx, sample_idx in enumerate(indices):
        if idx % 1000 == 0:
            if error_event.is_set() is True:
                raise RuntimeError()

        (path, target) = dataset[sample_idx]
        q_in[idx % len(q_in)].put((idx, path, target), block=True, timeout=None)

    for q in q_in:
        q.put(None, block=True, timeout=None)

    for p in read_processes:
        while True:
            p.join(timeout=2)
            if p.is_alive() is False:
                break

            if error_event.is_set() is True:
                raise RuntimeError()

    q_out.put(None, block=True, timeout=None)
    write_process.join()

    if error_event.is_set() is True:
        raise RuntimeError()

    if args.type == "wds":
        (wds_path, num_shards) = fs_ops.wds_braces_from_path(pack_path, prefix=f"{args.suffix}-{args.split}")
        logger.info(f"Packed {len(dataset):,} samples into {num_shards} shards at {wds_path}")
    elif args.type == "directory":
        logger.info(f"Packed {len(dataset):,} samples")

    toc = time.time()
    rate = len(dataset) / (toc - tic)
    (minutes, seconds) = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to pack {len(dataset):,} samples ({rate:.2f} samples/sec)")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "pack",
        allow_abbrev=False,
        help="pack image dataset",
        description="pack image dataset",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools pack --size 512 data/training\n"
            "python -m birder.tools pack -j 4 --shuffle --max-size 80 --target-path data/cub_200_2011 "
            "--suffix cub_200_2011 data/CUB_200_2011/training\n"
            "python -m birder.tools pack -j 4 --max-size 80 --target-path data/cub_200_2011 "
            "--suffix cub_200_2011 --split validation --append data/CUB_200_2011/validation\n"
            "python -m birder.tools pack --type directory -j 8 --suffix il-common_packed --size 448 "
            "--format jpeg --class-file data/il-common_classes.txt data/training\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--type", type=str, choices=["wds", "directory"], default="wds", help="pack type")
    subparser.add_argument("--target-path", type=str, help="where to write the packed dataset")
    subparser.add_argument("--max-size", type=int, default=400, help="maximum size of each shard in MB")
    subparser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="performs calculation on multiple cores, set -1 to run on all cores",
    )
    subparser.add_argument("--shuffle", default=False, action="store_true", help="shuffle the dataset during packing")
    subparser.add_argument("--size", type=int, help="resize image short dimension to size if bigger")
    subparser.add_argument("--format", type=str, choices=["webp", "png", "jpeg"], default="webp", help="file format")
    subparser.add_argument("--class-file", type=str, help="class list file")
    subparser.add_argument("--no-cls", default=False, action="store_true", help="pack without class information")
    subparser.add_argument("--suffix", type=str, default=settings.PACK_PATH_SUFFIX, help="directory suffix")
    subparser.add_argument("--split", type=str, default="training", help="dataset split used for _info.json")
    subparser.add_argument("--append", default=False, action="store_true", help="add split to existing wds")
    subparser.add_argument(
        "--sampling-file",
        type=str,
        help="file containing dataset paths and their sampling ratios (overrides data_path argument)",
    )
    subparser.add_argument("data_path", nargs="*", help="image directories")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.append is False or args.type == "wds"
    assert args.no_cls is False or args.type == "wds"
    assert args.sampling_file is None or len(args.data_path) == 0
    assert args.sampling_file is None or args.target_path is not None

    args.max_size = args.max_size * 1e6
    if args.target_path is None:
        pack_path = Path(f"{Path(args.data_path[0])}_{args.suffix}")
    else:
        pack_path = Path(args.target_path)

    if pack_path.exists() is False:
        logger.info(f"Creating {pack_path} directory...")
        pack_path.mkdir(parents=True)

    elif args.append is False:
        logger.warning("Directory already exists... aborting")
        raise SystemExit(1)

    pack(args, pack_path)
