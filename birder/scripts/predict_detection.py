import argparse
import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.datasets.directory import make_image_dataset
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.transforms.detection import batch_images
from birder.transforms.detection import inference_preset


# pylint: disable=too-many-locals
def predict(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
        logging.info(f"Using device {device}")

    (net, class_to_idx, signature, rgb_stats) = fs_ops.load_detection_model(
        device,
        args.network,
        net_param=args.net_param,
        config=args.model_config,
        tag=args.tag,
        reparameterized=args.reparameterized,
        backbone=args.backbone,
        backbone_param=args.backbone_param,
        backbone_config=args.backbone_model_config,
        backbone_tag=args.backbone_tag,
        backbone_reparameterized=args.backbone_reparameterized,
        epoch=args.epoch,
        new_size=args.size,
        quantized=args.quantized,
        inference=True,
        pts=args.pts,
        pt2=args.pt2,
        st=args.st,
    )

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.compile is True:
        net = torch.compile(net)
    elif args.compile_backbone is True:
        net.backbone.detection_features = torch.compile(net.backbone.detection_features)

    if args.parallel is True and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    if args.size is None:
        args.size = lib.get_size_from_signature(signature)[0]
        logging.debug(f"Using size={args.size}")

    score_threshold = args.min_score
    class_list = list(class_to_idx.keys())
    class_list.insert(0, "Background")

    # Set label colors
    cmap = plt.get_cmap("jet")
    color_list = []
    for c in np.linspace(0, 1, len(class_list)):
        rgb = cmap(c)[0:3]
        rgb = tuple(int(x * 255) for x in rgb)
        color_list.append(rgb)

    batch_size = args.batch_size
    dataset = make_image_dataset(args.data_path, {}, transforms=inference_preset(args.size, rgb_stats))
    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=8,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    tic = time.time()
    with tqdm(total=len(dataset), initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, _ in inference_loader:
            # Predict
            inputs = [i.to(device) for i in inputs]
            inputs = batch_images(inputs)
            with torch.amp.autocast(device.type, enabled=args.amp):
                (detections, _) = net(inputs)

            # Metrics
            # TBD

            # Show flags
            if args.show is True:
                (w, h) = inputs[0].shape[1:]
                for file_path, detection in zip(file_paths, detections):
                    scores = detection["scores"]
                    idxs = torch.where(scores > score_threshold)
                    scores = scores[idxs]
                    boxes = detection["boxes"][idxs]
                    labels = detection["labels"][idxs]
                    label_names = [f"{class_list[i]}: {s:.3f}" for i, s in zip(labels, scores)]
                    colors = [color_list[label] for label in labels]

                    img = decode_image(file_path)
                    (orig_w, orig_h) = img.shape[1:]
                    w_ratio = orig_w / w
                    h_ratio = orig_h / h
                    adjusted_boxes = boxes * torch.tensor([h_ratio, w_ratio, h_ratio, w_ratio]).to(device)

                    result_with_boxes = draw_bounding_boxes(
                        image=img,
                        boxes=adjusted_boxes,
                        labels=label_names,
                        colors=colors,
                        width=3,
                        font="DejaVuSans",
                        font_size=14,
                    )

                    fig = plt.figure(num=file_path, figsize=(12, 9))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.imshow(np.transpose(result_with_boxes, [1, 2, 0]))
                    ax.axis("off")

                    plt.tight_layout()
                    plt.show()

            # Update progress bar
            progress.update(n=batch_size)

    toc = time.time()
    rate = len(dataset) / (toc - tic)
    (minutes, seconds) = divmod(toc - tic, 60)
    logging.info(f"{int(minutes):0>2}m{seconds:04.1f}s to classify {len(dataset):,} samples ({rate:.2f} samples/sec)")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run detection prediction on directories and/or files",
        epilog=(
            "Usage example:\n"
            "python predict_detection.py --network faster_rcnn --backbone resnext_101 "
            "-e 0 data/detection_data/validation\n"
            "python predict_detection.py --network retinanet --backbone resnext_101 "
            "-e 0 --show --gpu --compile data/detection_data/training\n"
            "python predict_detection.py --network faster_rcnn --backbone resnext_101 "
            "-e 0 --min-score 0.25 --gpu --show --shuffle data/detection_data/validation\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the neural network to use (i.e. faster_rcnn)")
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required by some networks")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=registry.list_models(net_type=DetectorBackbone),
        help="the neural network to used as backbone",
    )
    parser.add_argument(
        "--backbone-param",
        type=float,
        help="network specific parameter, required by some networks (for the backbone)",
    )
    parser.add_argument(
        "--backbone-model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the backbone default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    parser.add_argument(
        "--backbone-reparameterized", default=False, action="store_true", help="load reparameterized backbone"
    )
    parser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    parser.add_argument("--quantized", default=False, action="store_true", help="load quantized model")
    parser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    parser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    parser.add_argument("--pts", default=False, action="store_true", help="load torchscript network")
    parser.add_argument("--pt2", default=False, action="store_true", help="load standardized model")
    parser.add_argument("--st", default=False, action="store_true", help="load Safetensors weights")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-backbone", default=False, action="store_true", help="enable backbone only compilation"
    )
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--min-score", type=float, default=0.5, help="prediction score threshold")
    parser.add_argument("--size", type=int, help="image size for inference (defaults to model signature)")
    parser.add_argument("--batch-size", type=int, default=8, metavar="N", help="the batch size")
    parser.add_argument("--show", default=False, action="store_true", help="show image predictions")
    parser.add_argument("--shuffle", default=False, action="store_true", help="predict samples in random order")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpu's")
    parser.add_argument("data_path", nargs="+", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.network is not None
    assert args.backbone is not None
    assert args.parallel is False or (args.parallel is True and args.gpu is True)
    assert args.parallel is False or args.compile is False
    assert args.compile is False or args.compile_backbone is False


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    args = argparse.Namespace(**kwargs)
    args = parser.parse_args([], args)
    validate_args(args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    if settings.RESULTS_DIR.exists() is False:
        logging.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    predict(args)


if __name__ == "__main__":
    main()
