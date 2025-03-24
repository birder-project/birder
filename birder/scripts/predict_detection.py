import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.utils import draw_bounding_boxes

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.datasets.coco import CocoInference
from birder.datasets.directory import make_image_dataset
from birder.inference.detection import infer_dataloader
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.results.detection import Results
from birder.transforms.detection import inference_preset

logger = logging.getLogger(__name__)


def save_output(
    output_path: Path, sample_paths: list[str], class_to_idx: dict[str, int], detections: list[dict[str, torch.Tensor]]
) -> None:
    detection_list = [{k: v.cpu().numpy().tolist() for k, v in detection.items()} for detection in detections]
    output = dict(zip(sample_paths, detection_list))
    output["class_to_idx"] = class_to_idx
    logger.info(f"Saving output at {output_path}")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


def show_detections(
    img_path: str,
    input_tensor: torch.Tensor,
    detection: dict[str, torch.Tensor],
    score_threshold: float,
    class_list: list[str],
    color_list: list[tuple[int, ...]],
    root_path: Path,
) -> None:
    (w, h) = input_tensor.shape[1:]
    scores = detection["scores"]
    idxs = torch.where(scores > score_threshold)
    scores = scores[idxs]
    boxes = detection["boxes"][idxs]
    labels = detection["labels"][idxs]
    label_names = [f"{class_list[i]}: {s:.3f}" for i, s in zip(labels, scores)]
    colors = [color_list[label] for label in labels]

    img = decode_image(root_path.joinpath(img_path))
    (orig_w, orig_h) = img.shape[1:]
    w_ratio = orig_w / w
    h_ratio = orig_h / h
    adjusted_boxes = boxes * torch.tensor([h_ratio, w_ratio, h_ratio, w_ratio]).to(input_tensor.device)

    if adjusted_boxes.size(0) == 0:
        result_with_boxes = img
    else:
        result_with_boxes = draw_bounding_boxes(
            image=img,
            boxes=adjusted_boxes,
            labels=label_names,
            colors=colors,
            width=3,
            font="DejaVuSans",
            font_size=14,
        )

    fig = plt.figure(num=img_path, figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.transpose(result_with_boxes, [1, 2, 0]))
    ax.axis("off")

    plt.tight_layout()
    plt.show()


# pylint: disable=too-many-locals,too-many-branches
def predict(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    elif args.mps is True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)

        logger.info(f"Using device {device}")

    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    amp_dtype: torch.dtype = getattr(torch, args.amp_dtype)
    network_name = lib.get_detection_network_name(
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        backbone=args.backbone,
        backbone_param=args.backbone_param,
        backbone_tag=args.backbone_tag,
    )
    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_detection_model(
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
        dtype=model_dtype,
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
        args.size = lib.get_size_from_signature(signature)
        logger.debug(f"Using size={args.size}")

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
    if args.coco_json_path is not None:
        labeled = True
        root_path = Path(args.data_path[0])
        dataset = CocoInference(root_path, args.coco_json_path, transforms=inference_preset(args.size, rgb_stats))
        if dataset.class_to_idx != class_to_idx:
            logger.warning("Dataset class to index differs from model")
    else:
        labeled = False
        root_path = Path("")
        dataset = make_image_dataset(args.data_path, {}, transforms=inference_preset(args.size, rgb_stats))

    num_samples = len(dataset)
    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=8,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    show_flag = args.show is True

    def batch_callback(
        file_paths: list[str],
        inputs: torch.Tensor,
        detections: list[dict[str, torch.Tensor]],
        _targets: list[dict[str, Any]],
    ) -> None:
        # Show flags
        if show_flag is True:
            for img_path, input_tensor, detection in zip(file_paths, inputs, detections):
                show_detections(
                    img_path,
                    input_tensor,
                    detection,
                    score_threshold=score_threshold,
                    class_list=class_list,
                    color_list=color_list,
                    root_path=root_path,
                )

    # Sort out output file names
    epoch_str = ""
    if args.epoch is not None:
        epoch_str = f"_e{args.epoch}"

    base_output_path = f"{network_name}_{len(class_to_idx)}{epoch_str}_{args.size[0]}px_{num_samples}"
    if args.model_dtype != "float32":
        base_output_path = f"{base_output_path}_{args.model_dtype}"
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    output_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.json")

    # Inference
    tic = time.time()
    with torch.inference_mode():
        (sample_paths, detections, targets) = infer_dataloader(
            device,
            net,
            inference_loader,
            model_dtype,
            args.amp,
            amp_dtype,
            num_samples,
            batch_callback=batch_callback,
        )

    toc = time.time()
    rate = len(dataset) / (toc - tic)
    (minutes, seconds) = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to classify {len(dataset):,} samples ({rate:.2f} samples/sec)")

    # Save output
    if args.save_output is True:
        save_output(output_path, sample_paths, class_to_idx, detections)

    # Handle results
    if labeled is True:
        results = Results(sample_paths, targets, detections, class_to_idx)
        if args.save_results is True:
            results.save(f"{base_output_path}.json")

        # Log short report

    else:
        if args.save_results is True:
            logger.warning("Annotations not provided, unable to save results")


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
            "python predict_detection.py --network faster_rcnn -t coco --backbone csp_resnet_50 "
            "--backbone-tag imagenet1k -e 0 --batch-size 1 --gpu --gpu-id 1 "
            "--coco-json-path data/detection_data/validation_annotations_coco.json data/detection_data"
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
    parser.add_argument("--st", "--safetensors", default=False, action="store_true", help="load Safetensors weights")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-backbone", default=False, action="store_true", help="enable backbone only compilation"
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--min-score", type=float, default=0.5, help="prediction score threshold")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    parser.add_argument("--batch-size", type=int, default=8, metavar="N", help="the batch size")
    parser.add_argument("--show", default=False, action="store_true", help="show image predictions")
    parser.add_argument("--shuffle", default=False, action="store_true", help="predict samples in random order")
    parser.add_argument("--save-results", default=False, action="store_true", help="save results object")
    parser.add_argument("--save-output", default=False, action="store_true", help="save raw output as CSV")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use (ignored in parallel mode)")
    parser.add_argument("--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpu's")
    parser.add_argument("--coco-json-path", type=str, help="COCO json path")
    parser.add_argument("data_path", nargs="+", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.network is not None
    assert args.backbone is not None
    assert args.parallel is False or (args.parallel is True and args.gpu is True)
    assert args.parallel is False or args.compile is False
    assert args.compile is False or args.compile_backbone is False
    assert args.amp is False or args.model_dtype == "float32"
    assert args.coco_json_path is None or len(args.data_path) == 1
    args.size = cli.parse_size(args.size)


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
        logger.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    predict(args)


if __name__ == "__main__":
    logger = logging.getLogger(__spec__.name)
    main()
