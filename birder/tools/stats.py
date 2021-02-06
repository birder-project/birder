import argparse
import json
import logging
import multiprocessing
import time
from collections import Counter
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from birder.common import util
from birder.conf import settings


def reduce_worker(queue) -> None:
    mean = np.zeros(3)
    mean_squares = np.zeros(3)

    while True:
        deq = queue.get()
        if deq is None:
            break

        mean += deq[0]
        mean_squares += deq[1]

    # Calculate standard deviation
    std = np.sqrt(mean_squares - np.square(mean))

    logging.info(f"mean_r: {mean[2] * 255}")
    logging.info(f"mean_g: {mean[1] * 255}")
    logging.info(f"mean_b: {mean[0] * 255}")
    logging.info(f"std_r: {std[2] * 255}")
    logging.info(f"std_g: {std[1] * 255}")
    logging.info(f"std_b: {std[0] * 255}")

    stats = {
        "mean_r": mean[2] * 255,
        "mean_g": mean[1] * 255,
        "mean_b": mean[0] * 255,
        "std_r": std[2] * 255,
        "std_g": std[1] * 255,
        "std_b": std[0] * 255,
        "scale": 1,
    }

    logging.info(f"Writing {settings.RGB_VALUES_FILENAME}")
    with open(settings.RGB_VALUES_FILENAME, "w") as handle:
        json.dump(stats, handle, indent=2)


def read_worker(q_in, q_out, num_samples: int) -> None:
    while True:
        deq = q_in.get()
        if deq is None:
            break

        (image_num, image_path) = deq
        if image_num % 5000 == 4999:
            logging.info(f"Calculating image no. {image_num + 1}...")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = img / 255.0
        mean = img.mean((0, 1)) / num_samples
        mean_squares = np.square(img).mean((0, 1)) / num_samples

        q_out.put((mean, mean_squares), block=True, timeout=None)


def img_stats(images_path: List[str], jobs: int):
    num_samples = len(images_path)

    q_in = []  # type: ignore
    for _ in range(jobs):
        q_in.append(multiprocessing.Queue(1024))

    q_out = multiprocessing.Queue(1024)  # type: ignore

    read_processes = []
    for idx in range(jobs):
        read_processes.append(
            multiprocessing.Process(target=read_worker, args=(q_in[idx], q_out, num_samples))
        )

    for p in read_processes:
        p.start()

    reduce_process = multiprocessing.Process(target=reduce_worker, args=(q_out,))
    reduce_process.start()

    for idx, image_path in enumerate(images_path):
        q_in[idx % len(q_in)].put((idx, image_path), block=True, timeout=None)

    for q in q_in:
        q.put(None, block=True, timeout=None)

    for p in read_processes:
        p.join()

    q_out.put(None, block=True, timeout=None)
    reduce_process.join()


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "stats", help="show image directory statistics", formatter_class=util.ArgumentHelpFormatter
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="performs calculation on multiple cores, set -1 to run on all cores, e.g. --jobs 4",
    )
    subparser.add_argument(
        "--class-graph",
        default=False,
        action="store_true",
        help="show class sample distribution",
    )
    subparser.add_argument(
        "--rgb",
        default=False,
        action="store_true",
        help="calculate RGB mean and std",
    )
    subparser.add_argument("--data-path", type=str, default=settings.DATA_DIR, help="image directory")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    inverse_classes = util.read_synset_reverse()
    labels = []
    images_path = []
    for image_path, label in util.list_images(args.data_path, False, skip_aug=True, skip_adv=True):
        labels.append(inverse_classes[label])
        images_path.append(image_path)

    if args.class_graph is True:
        # TODO: Move to core.gui
        label_count = Counter(labels)
        sorted_classes: List[str]
        sorted_count: List[int]
        (sorted_classes, sorted_count) = list(zip(*label_count.most_common()))  # type: ignore

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.barh(sorted_classes, sorted_count, color=plt.get_cmap("RdYlGn")(sorted_count))
        for i, count in enumerate(sorted_count):
            ax.text(
                count + 3,
                i,
                str(count),
                color="dimgrey",
                va="center",
                fontsize="small",
                fontweight="light",
                clip_on=True,
            )

        ax.set_title(
            f"{len(images_path):n} samples, {len(sorted_classes)} classes "
            f"({len(images_path) / len(sorted_classes):.0f} samples per class on average)"
        )

        plt.show()

    if args.rgb is True:
        jobs = args.jobs
        if jobs == -1:
            jobs = multiprocessing.cpu_count()

        logging.info(f"Running {jobs} read processes and 1 reduce process")

        tic = time.time()
        img_stats(images_path, jobs)
        toc = time.time()
        logging.info(f"Calculated {len(images_path)} images in {toc - tic:.1f}s")
