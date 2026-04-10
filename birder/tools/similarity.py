import argparse
import logging
import time
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from PIL import Image
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.data.datasets.directory import make_image_dataset
from birder.data.transforms.classification import inference_preset
from birder.inference import classification

logger = logging.getLogger(__name__)


def _pairwise_distance_metric(metric_name: str) -> str:
    if metric_name == "l2":
        return "euclidean"

    return metric_name


def _build_distance_df(sample_paths: list[str], distances: Any) -> pl.DataFrame:
    sample_1, sample_2 = list(zip(*combinations(sample_paths, 2)))
    return pl.DataFrame(
        {
            "sample_1": sample_1,
            "sample_2": sample_2,
            "distance": distances,
        }
    )


def similarity(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    elif args.mps is True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type) if args.amp is True else None
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    net, (class_to_idx, signature, rgb_stats, *_) = fs_ops.load_model(
        device,
        args.network,
        config=args.model_config,
        tag=args.tag,
        epoch=args.epoch,
        new_size=args.size,
        inference=True,
        reparameterized=args.reparameterized,
        dtype=model_dtype,
    )

    if args.channels_last is True:
        net = net.to(memory_format=torch.channels_last)
        logger.debug("Using channels-last memory format")

    if args.compile is True:
        net.embedding = torch.compile(net.embedding)

    if args.size is None:
        args.size = lib.get_size_from_signature(signature)
        logger.debug(f"Using size={args.size}")

    transform = inference_preset(args.size, rgb_stats, args.center_crop, args.simple_crop)
    dataset = make_image_dataset(args.data_path, class_to_idx, transforms=transform)
    num_samples = len(dataset)

    inference_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    tic = time.time()
    with torch.inference_mode():
        sample_paths, _outs, _labels, embedding_list = classification.infer_dataloader(
            device,
            net,
            inference_loader,
            return_embedding=True,
            channels_last=args.channels_last,
            model_dtype=model_dtype,
            amp=args.amp,
            amp_dtype=amp_dtype,
            num_samples=num_samples,
            **args.forward_kwargs,
        )

    embeddings = np.concatenate(embedding_list, axis=0)

    toc = time.time()
    rate = num_samples / (toc - tic)
    logger.info(f"{lib.format_duration(toc-tic)} to embed {num_samples:,} samples ({rate:.2f} samples/sec)")

    logger.info(f"Computing pairwise distances with metric={args.distance_metric}")
    distances = pdist(embeddings, metric=_pairwise_distance_metric(args.distance_metric))
    distance_df = _build_distance_df(sample_paths, distances).sort("distance", descending=args.reverse)

    limit = len(distance_df) if args.limit is None else min(args.limit, len(distance_df))
    for idx, pair in enumerate(distance_df[:limit].iter_rows(named=True), start=1):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(Image.open(pair["sample_1"]))
        ax1.set_title(pair["sample_1"])
        ax1.axis("off")

        ax2.imshow(Image.open(pair["sample_2"]))
        ax2.set_title(pair["sample_2"])
        ax2.axis("off")

        logger.info(f"{pair['distance']:.3f} distance between {pair['sample_1']} and {pair['sample_2']}")
        fig.suptitle(f"Distance = {pair['distance']:.3f} ({idx}/{len(distance_df):,})")
        plt.tight_layout()
        plt.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "similarity",
        allow_abbrev=False,
        help="show image pairs sorted by embedding similarity",
        description="show image pairs sorted by embedding similarity",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools similarity -n efficientnet_v1_b4 -e 300 --limit 3 data/*/Alpine\\ swift\n"
            "python -m birder.tools similarity -n rope_vit_reg4_b14 -t capi-raw336px -e 0 --gpu --gpu-id 1 "
            "--amp --batch-size 1 --simple-crop data/training/Sri\\ Lanka\\ frogmouth\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("-n", "--network", type=str, required=True, help="the neural network to use")
    subparser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    subparser.add_argument(
        "--forward-kwargs",
        action=cli.FlexibleDictAction,
        help=(
            "additional model forward/embedding keyword args, accepts key-value pairs or JSON "
            "('patch_size=12' or '{\"patch_size\": 20}'"
        ),
    )
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from the training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    subparser.add_argument(
        "--channels-last", default=False, action="store_true", help="use channels-last memory format"
    )
    subparser.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    subparser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    subparser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    subparser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    subparser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    subparser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    subparser.add_argument(
        "-j", "--num-workers", type=int, default=8, metavar="N", help="number of preprocessing workers"
    )
    subparser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    subparser.add_argument(
        "--simple-crop",
        default=False,
        action="store_true",
        help="use a simple crop that preserves aspect ratio but may trim parts of the image",
    )
    subparser.add_argument(
        "--distance-metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="distance metric to use for embedding comparison",
    )
    subparser.add_argument("--limit", type=int, metavar="N", help="limit number of pairs to show")
    subparser.add_argument("--reverse", default=False, action="store_true", help="start from most distinct pairs")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device"
    )
    subparser.add_argument("data_path", nargs="+", help="data files path (directories and files)")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    if args.forward_kwargs is None:
        args.forward_kwargs = {}

    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.batch_size < 1:
        raise cli.ValidationError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        raise cli.ValidationError(f"--num-workers must be >= 0, got {args.num_workers}")
    if args.prefetch_factor is not None and args.prefetch_factor < 1:
        raise cli.ValidationError(f"--prefetch-factor must be >= 1, got {args.prefetch_factor}")
    if args.prefetch_factor is not None and args.num_workers == 0:
        raise cli.ValidationError("--prefetch-factor requires --num-workers > 0")
    if args.limit is not None and args.limit < 1:
        raise cli.ValidationError(f"--limit must be >= 1, got {args.limit}")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    similarity(args)
