import argparse
import json
import logging
import multiprocessing
import os
import queue
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
from birder.common.lib import class_list_from_class_to_idx
from birder.common.lib import format_duration
from birder.conf import settings

logger = logging.getLogger(__name__)

# Few datasets like Objects365 have some very big files
Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 1024 // 4 // 3)
MAX_SIZE = 16_000
QUEUE_TIMEOUT = 1.0


class CustomImageFolder(ImageFolder):
    def __init__(self, root: str, *, class_to_idx: dict[str, int]) -> None:
        self._class_to_idx = class_to_idx
        super().__init__(root, loader=str, allow_empty=True)

    def find_classes(self, _directory: str) -> tuple[list[str], dict[str, int]]:
        classes = class_list_from_class_to_idx(self._class_to_idx)
        return (classes, self._class_to_idx)


def _get_class_to_idx(paths: list[str]) -> dict[str, int]:
    class_list: list[str] = []
    for path in paths:
        dataset = ImageFolder(path)
        class_list.extend(class_list_from_class_to_idx(dataset.class_to_idx))

    class_list = sorted(list(set(class_list)))
    class_to_idx = {k: v for v, k in enumerate(class_list)}

    return class_to_idx


def _save_classes(pack_path: Path, class_to_idx: dict[str, int]) -> None:
    class_list_path = pack_path.joinpath("classes.txt")
    doc = "\n".join(class_list_from_class_to_idx(class_to_idx))

    logger.info(f"Saving class list at {class_list_path}")
    with open(class_list_path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def _canonical_image_format(file_format: str) -> str:
    file_format = file_format.lower().removeprefix(".")
    if file_format == "jpg":
        return "jpeg"

    return file_format


def _encode_image(path: str, file_format: str, size: Optional[int] = None) -> bytes:
    file_format = _canonical_image_format(file_format)
    image: Image.Image
    with Image.open(path) as image:
        if file_format == "jpeg" and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

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


def _read_verified_image(path: str, file_format: str) -> bytes:
    file_format = _canonical_image_format(file_format)
    with open(path, "rb") as stream:
        sample = stream.read()

    with Image.open(BytesIO(sample)) as image:
        source_format = _canonical_image_format(image.format or "")
        # Intentionally avoid a full pixel decode to preserve same-format bytes and support very large images
        image.verify()

    if source_format != file_format:
        return _encode_image(path, file_format)

    return sample


def read_worker(q_in: Any, q_out: Any, error_event: Any, size: Optional[int], file_format: str) -> None:
    file_format = _canonical_image_format(file_format)
    while True:
        deq: Optional[tuple[int, str, int]] = q_in.get()
        if deq is None:
            break

        try:
            idx, path, target = deq
            if size is None:
                suffix = _canonical_image_format(Path(path).suffix)
                if file_format != suffix:
                    sample = _encode_image(path, file_format)
                else:
                    sample = _read_verified_image(path, file_format)

            else:
                sample = _encode_image(path, file_format, size)

        except Exception:
            error_event.set()
            logger.exception(f"Failed to read or encode image {path}")
            raise SystemExit(1) from None

        if error_event.is_set() is True:
            break

        while error_event.is_set() is False:
            try:
                q_out.put((idx, sample, file_format, target), block=True, timeout=QUEUE_TIMEOUT)
                break
            except queue.Full:
                continue


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

        filenames: list[str] = []
        shard_lengths: list[int] = []
        path_pattern = str(pack_path.joinpath(f"{args.suffix}-{args.split}-%06d.tar"))
        sink = wds.ShardWriter(path_pattern, maxsize=args.max_size, verbose=0)

        def wds_info(fname: str) -> None:
            filenames.append(Path(fname).name)
            shard_lengths.append(sink.count)

        sink.post = wds_info

        start_number = args.start_number
        count = 0
        buf = {}
        more = True
        with tqdm(total=total, initial=0, unit="images", unit_scale=True, leave=False) as progress:
            while more:
                deq: Optional[tuple[int, bytes, str, int]] = q_out.get()
                if deq is not None:
                    idx, sample, suffix, target = deq
                    buf[idx] = (sample, suffix, target)

                else:
                    more = False

                # Ensures ordered write
                while count in buf:
                    sample, suffix, target = buf[count]
                    del buf[count]

                    if args.no_cls is True:
                        cls = {}
                    else:
                        cls = {"cls": target}

                    sink.write(
                        {
                            "__key__": f"sample{count + start_number:09d}",
                            suffix: sample,
                            **cls,
                        }
                    )

                    count += 1

                    # Update progress bar
                    progress.update(n=1)

        if count != total or len(buf) != 0:
            raise RuntimeError(f"Writer received {count:,} of {total:,} samples")

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

    except Exception:
        error_event.set()
        logger.exception("WebDataset writer failed")
        raise SystemExit(1) from None


def directory_write_worker(
    q_out: Any, error_event: Any, pack_path: Path, total: int, args: argparse.Namespace, idx_to_class: dict[int, str]
) -> None:
    start_number = args.start_number
    count = 0
    buf = {}
    more = True
    try:
        with tqdm(total=total, initial=0, unit="images", unit_scale=True, leave=False) as progress:
            while more:
                deq: Optional[tuple[int, bytes, str, int]] = q_out.get()
                if deq is not None:
                    idx, sample, suffix, target = deq
                    buf[idx] = (sample, suffix, target)

                else:
                    more = False

                # Ensures ordered write
                while count in buf:
                    sample, suffix, target = buf[count]
                    del buf[count]
                    with open(
                        pack_path.joinpath(idx_to_class[target]).joinpath(f"{count + start_number:06d}.{suffix}"),
                        "wb",
                    ) as handle:
                        handle.write(sample)

                    count += 1

                    # Update progress bar
                    progress.update(n=1)

        if count != total or len(buf) != 0:
            raise RuntimeError(f"Writer received {count:,} of {total:,} samples")

    except Exception:
        error_event.set()
        logger.exception("Directory writer failed")
        raise SystemExit(1) from None


def pack(args: argparse.Namespace, pack_path: Path) -> None:
    if args.append is True:
        info = fs_ops.read_wds_info(pack_path.joinpath("_info.json"))
        if args.split in info["splits"]:
            raise ValueError(f"split {args.split} already exists")

    if args.sampling_file is not None:
        with open(args.sampling_file, "r", encoding="utf-8") as handle:
            sampling_lines = handle.readlines()

        data_paths = []
        for line in sampling_lines:
            if len(line.strip()) == 0 or line.strip().startswith("#") is True:
                continue

            data_path, r = line.split()
            data_path = os.path.expanduser(data_path)
            repeats = int(r)
            for _ in range(repeats):
                data_paths.append(data_path)

    else:
        data_paths = args.data_path

    class_to_idx: dict[str, int] = {}
    if args.no_cls is False:
        if args.append is True:
            class_list_path = pack_path.joinpath("classes.txt")
            if class_list_path.exists() is False:
                raise ValueError("cannot append classified samples to a pack without classes.txt")

            class_to_idx = fs_ops.read_class_file(class_list_path)
            if args.class_file is not None:
                append_class_to_idx = fs_ops.read_class_file(args.class_file)
                if append_class_to_idx != class_to_idx:
                    raise ValueError("class file does not match classes.txt in the target pack")

        elif args.class_file is not None:
            class_to_idx = fs_ops.read_class_file(args.class_file)
        else:
            class_to_idx = _get_class_to_idx(data_paths)

        if args.append is False:
            _save_classes(pack_path, class_to_idx)

        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

        datasets = []
        for path in data_paths:
            datasets.append(CustomImageFolder(path, class_to_idx=class_to_idx))

        dataset = ConcatDataset(datasets)

    else:
        if args.append is True and pack_path.joinpath("classes.txt").exists() is True:
            raise ValueError("cannot append samples without classes to a classified pack")

        idx_to_class = {}
        dataset = fs_ops.collect_samples_from_paths(data_paths, class_to_idx={})

    if args.shuffle is True:
        indices = torch.randperm(len(dataset)).tolist()
    else:
        indices = list(range(len(dataset)))

    if args.jobs == -1:
        jobs = multiprocessing.cpu_count()
    else:
        jobs = args.jobs

    if jobs < 1:
        raise ValueError("jobs must be a positive number or -1")

    logger.info(f"Packing {len(dataset):,} samples")
    logger.info(f"Running {jobs} read processes and 1 write process")

    if args.type == "wds":
        target_writer: Callable[..., None] = wds_write_worker
    elif args.type == "directory":
        target_writer = directory_write_worker
        for c in class_to_idx.keys():
            pack_path.joinpath(c).mkdir()
    else:
        raise ValueError("Unknown pack type")

    q_in = []  # type: ignore
    for _ in range(jobs):
        q_in.append(multiprocessing.Queue(1024))

    q_out = multiprocessing.Queue(1024)  # type: ignore
    error_event = multiprocessing.Event()

    read_processes: list[multiprocessing.Process] = []
    for idx in range(jobs):
        read_processes.append(
            multiprocessing.Process(
                name=f"pack-reader-{idx}",
                target=read_worker,
                args=(q_in[idx], q_out, error_event, args.size, args.format),
            )
        )

    write_process = multiprocessing.Process(
        name="pack-writer",
        target=target_writer,
        args=(q_out, error_event, pack_path, len(dataset), args, idx_to_class),
    )
    started_processes: list[multiprocessing.Process] = []

    # Flag to prevent signal handler re-entry
    cleanup_in_progress = False

    def cleanup_processes() -> None:
        nonlocal cleanup_in_progress
        if cleanup_in_progress is True:
            return

        cleanup_in_progress = True

        # Cancel queue join threads to prevent blocking during cleanup
        for q in q_in:
            q.cancel_join_thread()

        q_out.cancel_join_thread()

        # Terminate child processes
        for p in started_processes:
            if p.is_alive():
                p.terminate()

        # Wait briefly for termination
        deadline = time.monotonic() + 1
        for p in started_processes:
            p.join(timeout=max(0.0, deadline - time.monotonic()))

        # Ensure cleanup cannot leave a stuck child behind
        for p in started_processes:
            if p.is_alive():
                logger.warning(f"Killing unresponsive process {p.name}")
                p.kill()

        deadline = time.monotonic() + 1
        for p in started_processes:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                logger.error(f"Process {p.name} did not exit after being killed")

    def raise_process_failure(process: multiprocessing.Process, *, unexpected_exit: bool = False) -> None:
        error_event.set()
        if unexpected_exit is True and process.exitcode == 0:
            reason = "exited unexpectedly"
        else:
            reason = f"failed with exit code {process.exitcode}"

        logger.error(f"Pack process {process.name} {reason}")
        raise RuntimeError(f"Pack process {process.name} {reason}")

    def ensure_pipeline_healthy(completed_readers: set[int]) -> None:
        if error_event.is_set() is True:
            for process in started_processes:
                if process.exitcode not in (None, 0):
                    raise_process_failure(process)

            raise RuntimeError("A pack worker reported an error")

        for idx, process in enumerate(read_processes):
            if process.exitcode is None:
                continue
            if idx in completed_readers and process.exitcode == 0:
                continue

            raise_process_failure(process, unexpected_exit=True)

        if write_process.exitcode is not None:
            raise_process_failure(write_process, unexpected_exit=True)

    def put_with_health_check(q: Any, item: Any, completed_readers: set[int]) -> None:
        while True:
            if error_event.is_set() is True:
                ensure_pipeline_healthy(completed_readers)

            try:
                q.put(item, block=True, timeout=QUEUE_TIMEOUT)
                return
            except queue.Full:
                ensure_pipeline_healthy(completed_readers)

    def signal_handler(signum, _frame) -> None:  # type: ignore
        logger.info(f"Received signal: {signum} at {multiprocessing.current_process().name}, aborting...")
        error_event.set()
        cleanup_processes()
        raise SystemExit(1)

    previous_signal_handlers: dict[signal.Signals, Any] = {}

    try:
        for p in read_processes:
            p.start()
            started_processes.append(p)

        write_process.start()
        started_processes.append(write_process)

        # Install handlers after starting children so they do not inherit them
        for handled_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handler = signal.getsignal(handled_signal)
            signal.signal(handled_signal, signal_handler)
            previous_signal_handlers[handled_signal] = previous_handler

        tic = time.time()
        completed_readers: set[int] = set()
        for idx, sample_idx in enumerate(indices):
            path, target = dataset[sample_idx]
            put_with_health_check(q_in[idx % len(q_in)], (idx, path, target), completed_readers)

        ensure_pipeline_healthy(completed_readers)
        for idx, q in enumerate(q_in):
            put_with_health_check(q, None, completed_readers)
            completed_readers.add(idx)

        for p in read_processes:
            while p.is_alive():
                p.join(timeout=QUEUE_TIMEOUT)
                ensure_pipeline_healthy(completed_readers)

            if p.exitcode != 0:
                raise_process_failure(p)

        ensure_pipeline_healthy(completed_readers)
        put_with_health_check(q_out, None, completed_readers)
        while write_process.is_alive():
            write_process.join(timeout=QUEUE_TIMEOUT)
            if error_event.is_set() is True:
                raise RuntimeError("A pack worker reported an error")

        if write_process.exitcode != 0:
            raise_process_failure(write_process)

        if error_event.is_set() is True:
            raise RuntimeError("A pack worker reported an error")

        if args.type == "wds":
            wds_path, num_shards = fs_ops.wds_braces_from_path(pack_path, prefix=f"{args.suffix}-{args.split}")
            logger.info(f"Packed {len(dataset):,} samples into {num_shards} shards at {wds_path}")
        elif args.type == "directory":
            logger.info(f"Packed {len(dataset):,} samples")

        toc = time.time()
        rate = len(dataset) / (toc - tic)
        logger.info(f"{format_duration(toc-tic)} to pack {len(dataset):,} samples ({rate:.2f} samples/sec)")

    except BaseException:
        error_event.set()
        logger.error(f"Packing failed, output at {pack_path} may be incomplete")
        cleanup_processes()
        raise

    finally:
        for handled_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(handled_signal, previous_handler)

        for q in [*q_in, q_out]:
            q.close()
            if cleanup_in_progress is False:
                q.join_thread()


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
        "-j", "--jobs", type=int, default=1, help="performs calculation on multiple cores, set -1 to run on all cores"
    )
    subparser.add_argument("--shuffle", default=False, action="store_true", help="shuffle the dataset during packing")
    subparser.add_argument("--size", type=int, help="resize image short dimension to size if bigger")
    subparser.add_argument("--format", type=str, choices=["webp", "png", "jpeg"], default="webp", help="file format")
    subparser.add_argument("--class-file", type=str, help="class list file")
    subparser.add_argument("--no-cls", default=False, action="store_true", help="pack without class information")
    subparser.add_argument("--start-number", type=int, default=0, help="starting number for output file naming")
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
    if args.append is True and args.type != "wds":
        raise cli.ValidationError("--append requires --type wds to be set")
    if args.no_cls is True and args.type != "wds":
        raise cli.ValidationError("--no-cls requires --type wds to be set")

    if args.sampling_file is not None and len(args.data_path) > 0:
        raise cli.ValidationError("--sampling-file cannot be used with --data-path")
    if args.sampling_file is not None and args.target_path is None:
        raise cli.ValidationError("--sampling-file requires --target-path to be set")
    if args.sampling_file is None and len(args.data_path) == 0:
        raise cli.ValidationError("at least one data path is required")
    if args.jobs == 0 or args.jobs < -1:
        raise cli.ValidationError("--jobs must be a positive number or -1")

    args.max_size = args.max_size * 1e6
    if args.target_path is None:
        pack_path = Path(f"{Path(args.data_path[0])}_{args.suffix}")
    else:
        pack_path = Path(args.target_path)

    if args.append is True and pack_path.is_dir() is False:
        raise cli.ValidationError("--append requires an existing target directory")
    if pack_path.exists() is False:
        logger.info(f"Creating {pack_path} directory...")
        pack_path.mkdir(parents=True)

    elif args.append is False:
        logger.warning("Directory already exists... aborting")
        raise SystemExit(1)

    pack(args, pack_path)
