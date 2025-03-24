import argparse
import logging
from typing import Any

import torch
import torch.amp
from torch.utils.data import DataLoader

import birder
from birder.common import cli
from birder.conf import settings
from birder.datasets.directory import make_image_dataset

logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
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
    amp_dtype: torch.dtype = getattr(torch, args.amp_dtype)
    model_list = birder.list_pretrained_models(args.filter)
    for model_name in model_list:
        (net, (class_to_idx, signature, rgb_stats, *_)) = birder.load_pretrained_model(
            model_name, inference=True, device=device, dtype=model_dtype
        )
        if args.compile is True:
            net = torch.compile(net)

        if args.size is None:
            args.size = birder.get_size_from_signature(signature)

        transform = birder.classification_transform(args.size, rgb_stats, args.center_crop)
        dataset = make_image_dataset(args.data_path, class_to_idx, transforms=transform)
        num_samples = len(dataset)
        inference_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)
        with torch.inference_mode():
            results = birder.evaluate_classification(
                device,
                net,
                inference_loader,
                class_to_idx,
                args.tta,
                model_dtype,
                args.amp,
                amp_dtype,
                num_samples=num_samples,
            )

        logger.info(f"{model_name}: accuracy={results.accuracy:.3f}")
        base_output_path = (
            f"{args.dir}/{model_name}_{len(class_to_idx)}_{args.size[0]}px_crop{args.center_crop}_{num_samples}"
        )

        results.save(f"{base_output_path}.csv")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="evaluate pretrained models on specified dataset",
        epilog=(
            "Usage example:\n"
            "python evaluate.py --filter '*il-common*' --fast-matmul --gpu data/validation_il-common_packed\n"
            "python evaluate.py --compile --amp --gpu --gpu-id 1 data/testing\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("--filter", type=str, help="models to evaluate (fnmatch type filter)")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
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
    parser.add_argument("--tta", default=False, action="store_true", help="test time augmentation (oversampling)")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="the batch size")
    parser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    parser.add_argument(
        "--dir", type=str, default="evaluate", help="place all outputs in a sub-directory (relative to results)"
    )
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    parser.add_argument("--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device")
    parser.add_argument("data_path", nargs="+", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.amp is False or args.model_dtype == "float32"
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

    if settings.RESULTS_DIR.joinpath(args.dir).exists() is False:
        logger.info(f"Creating {settings.RESULTS_DIR.joinpath(args.dir)} directory...")
        settings.RESULTS_DIR.joinpath(args.dir).mkdir(parents=True)

    evaluate(args)


if __name__ == "__main__":
    logger = logging.getLogger(__spec__.name)
    main()
