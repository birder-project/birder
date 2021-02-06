import argparse
import logging
import multiprocessing
import random
import time

from birder.common import util
from birder.conf import settings
from birder.core.augmentations import worker


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "aug",
        help="augment images",
        description="Running aug multiple times may overwrite existing augmentations",
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--ratio",
        type=float,
        default=settings.AUG_RATIO,
        help="percent of images to augment",
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="performs calculation on multiple cores, set -1 to run on all cores, e.g. --jobs 4",
    )
    subparser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="show augmentations instead of writing them",
    )
    subparser.add_argument("--log-prefix", type=str, default="", help="log file prefix")
    subparser.add_argument("--data-path", type=str, default=settings.DATA_DIR, help="image directory")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.ratio <= 1.0 and args.ratio >= 0.0
    if args.visualize is True:
        assert args.jobs == 1, "Cannot visualize with more then one worker"

    if args.ratio == 0:
        logging.info("Ratio = 0, skipping augmentation")
        return

    jobs = args.jobs
    if jobs == -1:
        jobs = multiprocessing.cpu_count()

    logging.info(f"Running {jobs} processes")

    queues = []  # type: ignore
    for _ in range(jobs):
        queues.append(multiprocessing.Queue(4096))

    processes = []
    for idx in range(jobs):
        processes.append(multiprocessing.Process(target=worker, args=(queues[idx], args.visualize)))

    for p in processes:
        p.start()

    aug_count = 0
    tic = time.time()
    image_list = list(util.list_images(args.data_path, write_synset=False, skip_aug=True, skip_adv=True))
    for image_path, _ in image_list:
        if random.random() < args.ratio:
            queues[aug_count % len(queues)].put((aug_count, image_path), block=True, timeout=None)
            aug_count += 1

    for q in queues:
        q.put(None, block=True, timeout=None)

    for p in processes:
        p.join()

    util.write_logfile(args.log_prefix, "augmentation", {"ratio": args.ratio, "aug_count": aug_count})

    toc = time.time()
    rate = aug_count / (toc - tic)
    logging.info(f"Done, augmented {aug_count} images in {toc - tic:.2f}s ({rate:.2f} images/sec)")
