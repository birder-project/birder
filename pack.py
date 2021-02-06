import argparse
import logging
import multiprocessing
import os
import random
import time
from typing import Tuple

import cv2
import mxnet as mx

from birder.common import util
from birder.conf import settings

ImageTuple = Tuple[int, str, int]  # id, image path, label


def read_worker(q_in, q_out, size: int, pass_through: bool) -> None:
    while True:
        deq = q_in.get()
        if deq is None:
            break

        packed_s = image_encode(deq, (size, size), pass_through)
        q_out.put((deq[0], packed_s), block=True, timeout=None)


def write_worker(q_out, data_path: str) -> None:
    # Open file handler in the writing process
    rec_filename = os.path.join(data_path, settings.REC_FILENAME)
    idx_filename = os.path.join(data_path, settings.IDX_FILENAME)
    logging.info(f"Writing '{rec_filename}' and '{idx_filename}'")
    record = mx.recordio.MXIndexedRecordIO(idx_filename, rec_filename, "w")

    tic = time.time()
    count = 0
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            (idx, packed_s) = deq
            buf[idx] = packed_s

        else:
            more = False

        # Ensures ordered write
        while count in buf:
            packed_s = buf[count]
            del buf[count]
            if packed_s is not None:
                record.write_idx(count, packed_s)

            else:
                logging.warning(f"Empty image {count}")

            if count % 1000 == 999:
                logging.info(f"Packed {count + 1} images...")

            count += 1

    toc = time.time()
    rate = count / (toc - tic)
    logging.info(f"Packed {count} images in {toc - tic:.1f}s ({rate:.2f} images/sec)")


def image_encode(item: ImageTuple, reshape_size: Tuple[int, int], pass_through: bool) -> bytes:
    header = mx.recordio.IRHeader(flag=0, label=item[2], id=item[0], id2=0)

    if pass_through is True:
        with open(item[1], "rb") as fin:
            img = fin.read()

        packed = mx.recordio.pack(header, img)  # type: bytes

    else:
        img = cv2.imread(item[1], cv2.IMREAD_COLOR)
        img = cv2.resize(img, reshape_size, interpolation=cv2.INTER_CUBIC)
        packed = mx.recordio.pack_img(header, img, quality=8, img_fmt=".png")

    return packed


def pack_data(args: argparse.Namespace) -> None:
    image_list = list(util.list_images(args.data_path, args.write_synset))
    if args.only_write_synset is True:
        return

    if args.shuffle is True:
        random.shuffle(image_list)

    if args.jobs == -1:
        args.jobs = multiprocessing.cpu_count()

    logging.info(f"Running {args.jobs} read processes and 1 write process")

    q_in = []  # type: ignore
    for _ in range(args.jobs):
        q_in.append(multiprocessing.Queue(1024))

    q_out = multiprocessing.Queue(1024)  # type: ignore

    read_processes = []
    for idx in range(args.jobs):
        read_processes.append(
            multiprocessing.Process(target=read_worker, args=(q_in[idx], q_out, args.size, args.pass_through))
        )

    for p in read_processes:
        p.start()

    write_process = multiprocessing.Process(target=write_worker, args=(q_out, args.data_path))
    write_process.start()

    for idx, item in enumerate(image_list):
        q_in[idx % len(q_in)].put((idx,) + item, block=True, timeout=None)

    for q in q_in:
        q.put(None, block=True, timeout=None)

    for p in read_processes:
        p.join()

    q_out.put(None, block=True, timeout=None)
    write_process.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Packs data into an indexed rec file",
        epilog="Usage example:\npython3 pack.py --size 360 --jobs 4 --shuffle --write-synset data/",
        formatter_class=util.ArgumentHelpFormatter,
    )
    parser.add_argument("-c", "--write-synset", action="store_true", help="write synset.txt")
    parser.add_argument("--only-write-synset", action="store_true", help="write synset.txt and exit")
    parser.add_argument("--shuffle", action="store_true", help="randomize the image packing order")
    parser.add_argument("-s", "--size", type=int, default=360, help="image size to pack")
    parser.add_argument(
        "--pass-through", action="store_true", help="whether to skip transformation and save image as is"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="performs calculation on multiple cores, set -1 to run on all cores, e.g. --jobs 4",
    )
    parser.add_argument("data_path", help="image directory")
    args = parser.parse_args()

    os.makedirs(settings.MODELS_DIR, exist_ok=True)

    if args.only_write_synset is True:
        args.write_synset = True

    pack_data(args)


if __name__ == "__main__":
    main()
