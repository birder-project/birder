import argparse
import logging
import time
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from PIL import Image
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.datasets.directory import make_image_dataset
from birder.inference import classification
from birder.transforms.classification import inference_preset

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals
def similarity(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        inference=True,
        reparameterized=args.reparameterized,
    )

    size = lib.get_size_from_signature(signature)
    batch_size = 32
    dataset = make_image_dataset(args.data_path, class_to_idx, transforms=inference_preset(size, rgb_stats, 1.0))
    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    embedding_list: list[npt.NDArray[np.float32]] = []
    sample_paths: list[str] = []
    tic = time.time()
    with tqdm(total=len(dataset), initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, _targets in inference_loader:
            # Predict
            inputs = inputs.to(device)
            (_out, embedding) = classification.infer_batch(net, inputs, return_embedding=True)
            embedding_list.append(embedding)  # type: ignore[arg-type]
            sample_paths.extend(file_paths)

            # Update progress bar
            progress.update(n=batch_size)

    embeddings = np.concatenate(embedding_list, axis=0)

    toc = time.time()
    rate = len(dataset) / (toc - tic)
    (minutes, seconds) = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to classify {len(dataset)} samples ({rate:.2f} samples/sec)")

    logger.info("Processing similarity...")

    if args.cosine is True:
        distance_arr = pdist(embeddings, metric="cosine")

    else:
        # Dimensionality reduction
        tsne_embeddings_arr = TSNE(
            n_components=4, method="exact", learning_rate="auto", init="random", perplexity=5
        ).fit_transform(embeddings)

        # Build distance data frame
        distance_arr = distance_matrix(tsne_embeddings_arr, tsne_embeddings_arr)
        distance_arr = squareform(distance_arr)

    (sample_1, sample_2) = list(zip(*combinations(sample_paths, 2)))
    distance_df = pl.DataFrame(
        {
            "sample_1": sample_1,
            "sample_2": sample_2,
            "distance": distance_arr,
        }
    )
    distance_df = distance_df.sort("distance", descending=args.reverse)

    # Show image pairs
    if args.limit is None:
        args.limit = len(distance_df)

    for idx, pair in enumerate(distance_df[: args.limit].iter_rows(named=True)):
        (fig, (ax1, ax2)) = plt.subplots(2, 1)
        ax1.imshow(Image.open(pair["sample_1"]))
        ax1.set_title(pair["sample_1"])
        ax2.imshow(Image.open(pair["sample_2"]))
        ax2.set_title(pair["sample_2"])
        logger.info(f"{pair['distance']:.3f} distance between {pair['sample_1']} and {pair['sample_2']}")
        fig.suptitle(f"Distance = {pair['distance']:.3f} ({idx+1}/{len(distance_df):,})")
        plt.tight_layout()
        plt.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "similarity",
        allow_abbrev=False,
        help="show most similar images",
        description="show most similar images",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools similarity -n efficientnet_v1_b4 -e 300 data/*/Alpine\\ swift\n"
            "python -m birder.tools similarity -n efficientnet_v2_s -e 200 --limit 3 data/*/Arabian\\ babbler\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required by some networks"
    )
    subparser.add_argument("--cosine", default=False, action="store_true", help="use cosine distance")
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument("--limit", type=int, help="limit number of pairs to show")
    subparser.add_argument("--reverse", default=False, action="store_true", help="start from most distinct pairs")
    subparser.add_argument("data_path", nargs="+", help="data files path (directories and files)")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    similarity(args)
