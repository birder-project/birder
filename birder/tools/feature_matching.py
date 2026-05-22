import argparse
import logging
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.data.datasets.directory import pil_rgb_loader
from birder.data.transforms.classification import inference_preset

logger = logging.getLogger(__name__)

LINE_WIDTH = 2


def _extract_features(net: torch.nn.Module, inputs: torch.Tensor, stage: Optional[str]) -> tuple[str, torch.Tensor]:
    features_dict = net.detection_features(inputs)
    if stage is not None:
        features = features_dict[stage]
        stage_name = stage
    else:
        stage_name = list(features_dict.keys())[-1]
        features = list(features_dict.values())[-1]

    return (stage_name, features.float().cpu())


def _match_features(
    features1: torch.Tensor, features2: torch.Tensor, threshold: float, margin: float, limit: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[float]]:
    _, _, h1, w1 = features1.size()
    _, _, h2, w2 = features2.size()

    flat1 = F.normalize(features1.flatten(2).transpose(1, 2), dim=-1)
    flat2 = F.normalize(features2.flatten(2).transpose(1, 2), dim=-1)
    similarity = torch.matmul(flat1, flat2.transpose(-2, -1))[0]

    scores, matches2 = similarity.max(dim=1)
    matches1 = similarity.max(dim=0).indices
    idx1 = torch.arange(h1 * w1)

    keep = matches1[matches2] == idx1
    if margin > 0.0:
        if h2 * w2 > 1:
            top2 = torch.topk(similarity, k=2, dim=1).values
            keep &= top2[:, 0] - top2[:, 1] >= margin

        if h1 * w1 > 1:
            top2 = torch.topk(similarity, k=2, dim=0).values
            keep &= top2[0, matches2] - top2[1, matches2] >= margin

    if keep.any().item() is True:
        threshold_similarity = float(scores[keep].max().item())
    else:
        threshold_similarity = None

    keep &= scores >= threshold
    idx1 = idx1[keep]
    idx2 = matches2[keep]
    scores = scores[keep]
    if len(scores) == 0:
        return (idx1, idx2, scores, threshold_similarity)

    order = scores.argsort(descending=True)
    if limit is not None:
        order = order[:limit]

    return (idx1[order], idx2[order], scores[order], threshold_similarity)


def _resize_size(image_size: tuple[int, int], size: int | tuple[int, int]) -> tuple[int, int]:
    image_w, image_h = image_size
    if isinstance(size, int):
        short = min(image_w, image_h)
        long = max(image_w, image_h)
        new_short = size
        new_long = int(size * long / short)
        if image_w <= image_h:
            return (new_short, new_long)

        return (new_long, new_short)

    return (size[1], size[0])


def _feature_region(
    image_size: tuple[int, int],
    size: tuple[int, int],
    center_crop: float,
    simple_crop: bool,
) -> tuple[float, float, float, float]:
    if simple_crop is True:
        base_size: int | tuple[int, int] = int(min(size) / center_crop)
    else:
        base_size = (int(size[0] / center_crop), int(size[1] / center_crop))

    resized_w, resized_h = _resize_size(image_size, base_size)
    crop_top = round((resized_h - size[0]) / 2.0)
    crop_left = round((resized_w - size[1]) / 2.0)
    scale_x = resized_w / image_size[0]
    scale_y = resized_h / image_size[1]

    x0 = crop_left / scale_x
    y0 = crop_top / scale_y
    x1 = (crop_left + size[1]) / scale_x
    y1 = (crop_top + size[0]) / scale_y

    return (x0, y0, x1, y1)


def _cell_box(
    index: int, grid_size: tuple[int, int], region: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    grid_h, grid_w = grid_size
    region_x0, region_y0, region_x1, region_y1 = region
    region_w = region_x1 - region_x0
    region_h = region_y1 - region_y0
    row = index // grid_w
    col = index % grid_w

    x0 = round(region_x0 + col * region_w / grid_w)
    y0 = round(region_y0 + row * region_h / grid_h)
    x1 = round(region_x0 + (col + 1) * region_w / grid_w)
    y1 = round(region_y0 + (row + 1) * region_h / grid_h)

    return (x0, y0, x1, y1)


def _cell_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _draw_matches(
    image1: Image.Image,
    image2: Image.Image,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    grid_size1: tuple[int, int],
    grid_size2: tuple[int, int],
    region1: tuple[float, float, float, float],
    region2: tuple[float, float, float, float],
) -> Image.Image:
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")
    w1, h1 = image1.size
    w2, h2 = image2.size

    result = Image.new("RGB", (w1 + w2, max(h1, h2)), color=(0, 0, 0))
    result.paste(image1, (0, 0))
    result.paste(image2, (w1, 0))

    draw = ImageDraw.Draw(result)
    rng = np.random.default_rng(0)
    for match_idx1, match_idx2 in zip(idx1.tolist(), idx2.tolist(), strict=True):
        color = tuple(int(c) for c in rng.integers(32, 256, size=3))
        box1 = _cell_box(match_idx1, grid_size1, region1)
        box2 = _cell_box(match_idx2, grid_size2, region2)
        x1, y1 = _cell_center(box1)
        x2, y2 = _cell_center(box2)

        draw.line((x1, y1, x2 + w1, y2), fill=color, width=LINE_WIDTH)
        draw.rectangle(box1, outline=color, width=LINE_WIDTH)
        draw.rectangle((box2[0] + w1, box2[1], box2[2] + w1, box2[3]), outline=color, width=LINE_WIDTH)

    return result


def feature_matching(args: argparse.Namespace) -> None:
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

    net, model_info = fs_ops.load_model(
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

    if args.size is None:
        args.size = lib.get_size_from_signature(model_info.signature)
        logger.debug(f"Using size={args.size}")

    input_channels = lib.get_channels_from_signature(model_info.signature)
    if input_channels != 3:
        raise RuntimeError(f"feature matching expects RGB models, got input_channels={input_channels}")

    transform = inference_preset(args.size, model_info.rgb_stats, args.center_crop, args.simple_crop)
    image1 = pil_rgb_loader(args.image_path1)
    image2 = pil_rgb_loader(args.image_path2)

    input1 = transform(image1)
    input2 = transform(image2)
    region1 = _feature_region(image1.size, args.size, args.center_crop, args.simple_crop)
    region2 = _feature_region(image2.size, args.size, args.center_crop, args.simple_crop)

    input1 = input1.unsqueeze(0).to(device, dtype=model_dtype)
    input2 = input2.unsqueeze(0).to(device, dtype=model_dtype)

    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            stage1, features1 = _extract_features(net, input1, args.stage)
            stage2, features2 = _extract_features(net, input2, args.stage)

    idx1, idx2, scores, threshold_similarity = _match_features(
        features1, features2, args.threshold, args.margin, args.limit
    )
    grids = (
        f"{stage1} grid {features1.size(-2)}x{features1.size(-1)} "
        f"and {stage2} grid {features2.size(-2)}x{features2.size(-1)}"
    )
    if len(scores) > 0:
        logger.info(
            f"Found {len(scores):,} matches between {grids}, "
            f"with similarity ranging from {scores[-1].item():.3f} to {scores[0].item():.3f}"
        )
    elif threshold_similarity is not None:
        logger.info(f"Found 0 matches between {grids}, closest feature pair has similarity {threshold_similarity:.3f}")
    else:
        logger.info(f"Found 0 matches between {grids}, no mutual matches satisfy the current margin={args.margin:.3f}")

    result = _draw_matches(
        image1,
        image2,
        idx1,
        idx2,
        (features1.size(-2), features1.size(-1)),
        (features2.size(-2), features2.size(-1)),
        region1,
        region2,
    )

    plt.imshow(result)
    plt.title(f"{len(scores)} matches")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "feature-matching",
        allow_abbrev=False,
        help="show dense feature correspondences between two images",
        description="show dense feature correspondences between two images",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools feature-matching -n vit_reg4_b16 -t dino-v2-bio --gpu "
            "--threshold 0.65 image1.jpeg image2.jpeg\n"
            "python -m birder.tools feature-matching -n convnext_v2_tiny -e 0 --stage stage4 "
            "--limit 5 image1.jpeg image2.jpeg\n"
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
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from the training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
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
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    subparser.add_argument(
        "--simple-crop",
        default=False,
        action="store_true",
        help="use a simple crop that preserves aspect ratio but may trim parts of the image",
    )
    subparser.add_argument("--stage", type=str, help="feature stage to match, defaults to the last returned stage")
    subparser.add_argument("--threshold", type=float, default=0.7, help="minimum mutual-match cosine similarity")
    subparser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="minimum best-vs-second-best cosine similarity margin in both match directions",
    )
    subparser.add_argument("--limit", type=int, metavar="N", help="show only the top N matching features")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device"
    )
    subparser.add_argument("image_path1", type=str, help="first input image path")
    subparser.add_argument("image_path2", type=str, help="second input image path")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)

    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.threshold < -1 or args.threshold > 1:
        raise cli.ValidationError(f"--threshold must be in range of [-1, 1], got {args.threshold}")
    if args.margin < 0 or args.margin > 2:
        raise cli.ValidationError(f"--margin must be in range of [0, 2], got {args.margin}")
    if args.limit is not None and args.limit < 1:
        raise cli.ValidationError(f"--limit must be >= 1, got {args.limit}")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    feature_matching(args)
