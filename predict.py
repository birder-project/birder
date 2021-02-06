import argparse
import logging
import os
import random
import time
from collections import namedtuple
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd

from birder.common import util
from birder.common.net import create_model
from birder.common.preprocess import preprocess_image
from birder.conf import settings
from birder.core.gui import ConfusionMatrix
from birder.core.results import Results

Batch = namedtuple("Batch", ["data"])


def show_topk(
    orig_img: mx.nd.NDArray, image_path: str, out: np.ndarray, top_k: int, classes: Dict[str, int]
) -> None:
    inverse_classes = dict(zip(classes.values(), classes.keys()))
    probabilities = []
    predicted_class_names = []

    logging.info(f"'{image_path}'")
    for idx in np.argsort(out)[::-1][:top_k]:
        probabilities.append(out[idx])
        predicted_class_names.append(inverse_classes[idx])
        logging.info(f"{inverse_classes[idx]:<25}: {out[idx]:.4f}")

    logging.info("---")

    fig = plt.figure(num=os.path.basename(image_path))

    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(orig_img.asnumpy())
    ax.axis("off")

    ax = fig.add_subplot(2, 1, 2)
    y_pos = np.arange(top_k)
    bars = ax.barh(y_pos, probabilities, alpha=0.4, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predicted_class_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)

    label = util.get_label_from_path(image_path)
    for idx, class_name in enumerate(predicted_class_names):
        if label == class_name:
            bars[idx].set_color("green")

    plt.tight_layout()
    plt.show()


def save_output(image_list: List[str], output: List[np.ndarray], path: str) -> None:
    output_df = pd.DataFrame(output, columns=np.arange(len(output[0])).tolist())
    output_df.insert(0, "image", image_list)

    results_path = os.path.join(Results.results_dir, path)
    logging.info(f"Saving output at {results_path}")
    os.makedirs(Results.results_dir, exist_ok=True)

    # Write the dataframe
    output_df.to_csv(results_path, index=False, mode="w")


def predict(
    model: mx.module.Module,
    classes: Dict[str, int],
    rgb_mean: mx.nd.NDArray,
    rgb_std: mx.nd.NDArray,
    rgb_scale: float,
    args: argparse.Namespace,
) -> None:
    tic = time.time()
    labels: List[int] = []
    outs: List[np.ndarray] = []
    features: List[np.ndarray] = []
    for idx, image_path in enumerate(args.image_path):
        if idx % 1000 == 999:
            logging.info(f"Predicting sample {idx + 1}/{len(args.image_path)}")

        # Load image
        orig_img = mx.image.imread(image_path, flag=1, to_rgb=True)
        img = preprocess_image(
            orig_img, (args.size, args.size), rgb_mean, rgb_std, rgb_scale, center_crop=args.center_crop
        )

        # Predict
        model.forward(Batch([img]), is_train=False)
        out = np.squeeze(model.get_outputs()[0].asnumpy())
        if args.save_features is True:
            feature = np.squeeze(model.get_outputs()[1].asnumpy())
            features.append(feature)

        # Set prediction and labels (if exist)
        label = util.get_label_from_path(image_path)
        if label in classes:
            labels.append(classes[label])

        else:
            labels.append(-1)

        outs.append(out)

        # Show prediction
        if args.show is True:
            show_topk(orig_img, image_path, out, args.k, classes)

        # Show mistake (if label exists)
        if labels[-1] != -1:
            if args.show_mistakes is True:
                if np.argsort(out)[::-1][0] != labels[-1]:
                    show_topk(orig_img, image_path, out, args.k, classes)

            elif args.show_out_of_k is True:
                if labels[-1] not in np.argsort(out)[::-1][0 : args.k]:
                    show_topk(orig_img, image_path, out, args.k, classes)

    toc = time.time()
    if args.show is False and args.show_mistakes is False:
        rate = len(outs) / (toc - tic)
        logging.info(f"Took {toc - tic:.2f}s to classify {len(outs)} samples ({rate:.2f} samples/sec)")

    results = Results(args.image_path, labels, list(classes.keys()), outs)

    if args.save_output is True:
        save_output(
            args.image_path,
            outs,
            f"{args.network}_{len(classes)}_e{args.epoch}_{args.size}"
            f"px_crop{args.center_crop}_{len(results)}_output.csv",
        )

    if args.save_features is True:
        save_output(
            args.image_path,
            features,
            f"{args.network}_{len(classes)}_e{args.epoch}_{args.size}"
            f"px_crop{args.center_crop}_{len(results)}_features.csv",
        )

    # All predictions have labels
    if results.missing_labels is False:
        results.log_short_report()

        # Show confusion matrix
        if args.show_cnf is True:
            ConfusionMatrix(results).show()

        # Save results
        if args.save_results is True:
            results.save(
                f"{args.network}_{len(classes)}_e{args.epoch}_"
                f"{args.size}px_crop{args.center_crop}_{len(results)}.csv"
            )

    else:
        logging.warning("Some samples were missing labels")


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Predict image",
        epilog=(
            "Usage example:\n"
            "python3 predict.py --network resnet_v2_50 val_data/*/000001.jpeg\n"
            "python3 predict.py --network shufflenet_v1_1 --size 224 --epoch 100 --gpu val_data/*/*.jpeg"
        ),
        formatter_class=util.ArgumentHelpFormatter,
    )
    parser.add_argument(
        "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2_50)"
    )
    parser.add_argument(
        "--size", type=int, default=None, help="image size for inference (defaults to model signature)"
    )
    parser.add_argument("-e", "--epoch", type=int, default=0, help="model checkpoint to load")
    parser.add_argument("--k", type=int, default=settings.TOP_K, help="top K")
    parser.add_argument("--center-crop", type=float, default=1.0, help="Center crop ratio during inference")
    parser.add_argument(
        "--show", default=False, action="store_true", help="show image with top K predictions"
    )
    parser.add_argument(
        "--show-mistakes", default=False, action="store_true", help="show only mis-classified images"
    )
    parser.add_argument(
        "--show-out-of-k", default=False, action="store_true", help="show only images not in top-k"
    )
    parser.add_argument("--show-cnf", default=False, action="store_true", help="show confusion matrix")
    parser.add_argument(
        "--shuffle", default=False, action="store_true", help="go over predictions in random order"
    )
    parser.add_argument(
        "--save-features", default=False, action="store_true", help="save features layer output as CSV"
    )
    parser.add_argument("--save-output", default=False, action="store_true", help="save raw output as CSV")
    parser.add_argument("--save-results", default=False, action="store_true", help="save results as CSV")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu instead of cpu")
    parser.add_argument("image_path", nargs="+", help="image file path or rec file path")
    args = parser.parse_args()

    assert args.show is False or args.show_mistakes is False

    # Load synset
    classes = util.read_synset()

    # Load model
    if args.gpu is True:
        context = mx.gpu()

    else:
        context = mx.cpu()

    network_name = f"{args.network}_{len(classes)}"
    signature = util.read_signature(network_name)

    # Set size
    if args.size is None:
        args.size = util.get_signature_size(signature)
        logging.debug(f"Using size={args.size}")

    (sym, arg_params, aux_params) = mx.model.load_checkpoint(util.get_model_path(network_name), args.epoch)
    model = create_model(
        sym,
        arg_params,
        aux_params,
        context,
        (args.size, args.size),
        for_training=False,
        features=args.save_features,
    )

    # Set RGB
    rgb_mean = util.get_rgb_mean(signature)
    rgb_std = util.get_rgb_std(signature)
    rgb_scale = signature["scale"]

    # Shuffle images
    if args.shuffle is True:
        random.shuffle(args.image_path)

    predict(model, classes, rgb_mean, rgb_std, rgb_scale, args)


if __name__ == "__main__":
    main()
