import argparse

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

from birder.common import util
from birder.conf import settings
from birder.core import iterators

ITERATORS = {
    "ImageRecordIter": mx.io.ImageRecordIter,
    "ImageRecordIterMixup": iterators.ImageRecordIterMixup,
}


def show_iterator(iterator):
    for batch_num, batch in enumerate(iterator):
        for images in batch.data:
            num_cols = len(images) // 2
            num_rows = int(np.ceil(len(images) / num_cols))
            (fig, ax) = plt.subplots(num_rows, num_cols)
            ax = ax.ravel()
            for idx, img in enumerate(images):
                img = img.asnumpy()
                img = np.squeeze(img)
                img = np.uint8(img)
                img = img.transpose((1, 2, 0))

                ax[idx].imshow(img)
                ax[idx].axis("off")

            fig.suptitle(f"{type(iterator).__name__} - batch: {batch_num}")
            plt.show()


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "show-iterator",
        help="show images from rec file using image iterators",
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--iterator", type=str, choices=ITERATORS.keys(), required=True, help="the type of iterator to use"
    )
    subparser.add_argument("--rec-path", type=str, default=settings.DATA_PATH, help="rec file")
    subparser.add_argument("--size", type=int, default=224, help="image output size")
    subparser.add_argument("--max-aspect-ratio", type=float, default=4.0 / 3.0, help="maximum aspect ratio")
    subparser.add_argument("--min-aspect-ratio", type=float, default=3.0 / 4.0, help="minimum aspect ratio")
    subparser.add_argument("--max-random-area", type=float, default=1.0, help="maximum random area")
    subparser.add_argument("--min-random-area", type=float, default=0.7, help="minimum random area")
    subparser.add_argument("--brightness", type=float, default=0.3, help="brightness jitter")
    subparser.add_argument("--contrast", type=float, default=0.3, help="contrast jitter")
    subparser.add_argument("--saturation", type=float, default=0.3, help="saturation jitter")
    subparser.add_argument("--pca-noise", type=float, default=0.1, help="PCA based noise")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    classes = util.read_synset()
    batch_size = 8
    kwargs = {
        "path_imgrec": args.rec_path,
        "label_width": 1,
        "data_shape": (3, args.size, args.size),
        "batch_size": batch_size,
        "preprocess_threads": 1,
        "verbose": False,
        "random_resized_crop": True,
        "dtype": "float32",
        "max_aspect_ratio": args.max_aspect_ratio,
        "min_aspect_ratio": args.min_aspect_ratio,
        "max_random_area": args.max_random_area,
        "min_random_area": args.min_random_area,
        "brightness": args.brightness,
        "contrast": args.contrast,
        "saturation": args.saturation,
        "pca_noise": args.pca_noise,
        "fill_value": 127,
        "inter_method": 2,
        "rand_mirror": True,
    }

    iterator = ITERATORS[args.iterator](num_classes=len(classes), **kwargs)

    show_iterator(iterator)
