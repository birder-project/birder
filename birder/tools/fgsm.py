import argparse
import logging
import multiprocessing
import random
import time

import mxnet as mx

from birder.common import util
from birder.common.net import create_model
from birder.conf import settings
from birder.core.adversarial import FGSM
from birder.core.adversarial import write_worker


def _get_model(args: argparse.Namespace, network_name: str) -> mx.module.Module:
    (sym, arg_params, aux_params) = mx.model.load_checkpoint(util.get_model_path(network_name), args.epoch)
    model = create_model(
        sym,
        arg_params,
        aux_params,
        mx.cpu(),
        (args.size, args.size),
        for_training=True,
        inputs_need_grad=True,
    )

    return model


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "fgsm",
        help="generate fgsm adversarial images",
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--network", type=str, required=True, help="the neural network to use (i.e. mobilenet_v1_1.0)"
    )
    subparser.add_argument(
        "--size", type=int, default=None, help="network image size (defaults to model signature)"
    )
    subparser.add_argument("-e", "--epoch", type=int, default=0, help="model checkpoint to load")
    subparser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help="percent of images to generate",
    )
    subparser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="show samples instead of writing them",
    )
    subparser.add_argument("--log-prefix", type=str, default="", help="log file prefix")
    subparser.add_argument("--data-path", type=str, default=settings.DATA_DIR, help="image directory")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.ratio <= 1.0 and args.ratio >= 0.0

    if args.ratio == 0:
        logging.info("Ratio = 0, skipping fgsm")
        return

    # Load synset
    classes = util.read_synset()

    network_name = f"{args.network}_{len(classes)}"
    signature = util.read_signature(network_name)

    # Set size
    if args.size is None:
        args.size = util.get_signature_size(signature)
        logging.debug(f"Using size={args.size}")

    # Set RGB
    rgb_mean = util.get_rgb_mean(signature)
    rgb_std = util.get_rgb_std(signature)
    rgb_scale = signature["scale"]

    model = _get_model(args, network_name)

    fgsm = FGSM(rgb_mean, rgb_std, rgb_scale)

    queue = multiprocessing.Queue(4096)  # type: ignore
    write_process = multiprocessing.Process(target=write_worker, args=(queue,))
    write_process.start()

    adv_count = 0
    tic = time.time()
    image_list = list(util.list_images(args.data_path, write_synset=False, skip_aug=True, skip_adv=True))
    for image_path, _ in image_list:
        if random.random() < args.ratio:
            img = mx.image.imread(image_path, flag=1, to_rgb=True)
            label = mx.nd.array([classes[util.get_label_from_path(image_path)]])
            adv_img = fgsm(model, img, label, args.size)
            if args.visualize is True:
                fgsm.visualize(img, adv_img)

            else:
                queue.put((adv_count, image_path, adv_img), block=True, timeout=None)

            adv_count += 1

    queue.put(None, block=True, timeout=None)
    write_process.join()

    util.write_logfile(
        args.log_prefix,
        "fgsm",
        {
            "adv_count": adv_count,
            "epoch": args.epoch,
            "network": args.network,
            "ratio": args.ratio,
            "size": args.size,
        },
    )

    toc = time.time()
    logging.info(f"Done, generated {adv_count} images in {toc - tic:.2f}s")
