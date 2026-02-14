import argparse
import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

import birder
from birder.common import cli
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.inference.data_parallel import InferenceDataParallel

logger = logging.getLogger(__name__)


# pylint: disable=too-many-branches
def evaluate(args: argparse.Namespace) -> None:
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

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    amp_dtype: torch.dtype = getattr(torch, args.amp_dtype)
    model_list = birder.list_pretrained_models(args.filter)
    for model_name in model_list:
        net, (class_to_idx, signature, rgb_stats, *_) = birder.load_pretrained_model(
            model_name, inference=True, device=device, dtype=model_dtype
        )
        if args.channels_last is True:
            net = net.to(memory_format=torch.channels_last)
            logger.debug("Using channels-last memory format")

        if args.parallel is True and torch.cuda.device_count() > 1:
            net = InferenceDataParallel(net, output_device="cpu", compile_replicas=args.compile)
        elif args.compile is True:
            net = torch.compile(net)

        if args.size is None:
            size = birder.get_size_from_signature(signature)
        else:
            size = args.size

        transform = birder.classification_transform(size, rgb_stats, args.center_crop, args.simple_crop)

        if args.wds is True:
            wds_path: str | list[str]
            if args.wds_info is not None:
                wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
                if args.wds_size is not None:
                    dataset_size = args.wds_size
            else:
                wds_path, dataset_size = prepare_wds_args(args.data_path[0], args.wds_size, device)

            num_samples = dataset_size
            dataset = make_wds_dataset(
                wds_path,
                dataset_size=dataset_size,
                shuffle=False,
                samples_names=True,
                transform=transform,
            )
            inference_loader = make_wds_loader(
                dataset,
                args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=None,
                collate_fn=None,
                world_size=1,
                pin_memory=False,
                exact=True,
            )
        else:
            dataset = make_image_dataset(args.data_path, class_to_idx, transforms=transform)
            num_samples = len(dataset)
            inference_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        with torch.inference_mode():
            results = birder.evaluate_classification(
                device,
                net,
                inference_loader,
                class_to_idx,
                args.tta,
                args.channels_last,
                model_dtype,
                args.amp,
                amp_dtype,
                num_samples=num_samples,
                sparse=args.save_sparse_results,
            )

        logger.info(f"{model_name}: accuracy={results.accuracy:.4f}")
        base_output_path = (
            f"{args.dir}/{model_name}_{len(class_to_idx)}_{size[0]}px_crop{args.center_crop}_{num_samples}"
        )
        if args.save_sparse_results is True:
            results_file_suffix = "_sparse.csv"
        else:
            results_file_suffix = ".csv"

        results.save(f"{base_output_path}{results_file_suffix}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "classification",
        allow_abbrev=False,
        help="evaluate pretrained classification models on a dataset",
        description="evaluate pretrained classification models on a dataset",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval classification --filter '*il-all*' --fast-matmul --gpu "
            "data/validation_il-all_packed\n"
            "python -m birder.eval classification --amp --compile --gpu --gpu-id 1 data/testing\n"
            "python -m birder.eval classification --filter '*inat21*' --amp --compile --gpu "
            "--parallel ~/Datasets/inat2021/val\n"
            "python -m birder.eval classification --wds --wds-info data/validation_wds/info.json --gpu\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--filter", type=str, help="models to evaluate (fnmatch type filter)")
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
        default="float16",
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    subparser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    subparser.add_argument("--tta", default=False, action="store_true", help="test time augmentation (oversampling)")
    subparser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    subparser.add_argument("--batch-size", type=int, default=64, metavar="N", help="the batch size")
    subparser.add_argument(
        "-j", "--num-workers", type=int, default=8, metavar="N", help="number of preprocessing workers"
    )
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    subparser.add_argument(
        "--simple-crop",
        default=False,
        action="store_true",
        help="use a simple crop that preserves aspect ratio but may trim parts of the image",
    )
    subparser.add_argument(
        "--dir", type=str, default="evaluate", help="place all outputs in a sub-directory (relative to results)"
    )
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device"
    )
    subparser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpus")
    subparser.add_argument(
        "--save-sparse-results",
        default=False,
        action="store_true",
        help="save results object in memory-efficient sparse format (only top-k probabilities)",
    )
    subparser.add_argument("--wds", default=False, action="store_true", help="evaluate a webdataset directory")
    subparser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds dataset")
    subparser.add_argument("--wds-info", type=str, metavar="FILE", help="wds info file path")
    subparser.add_argument(
        "--wds-split", type=str, default="validation", metavar="NAME", help="wds dataset split to load"
    )
    subparser.add_argument("data_path", nargs="*", help="data files path (directories and files)")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.parallel is True and args.gpu is False:
        raise cli.ValidationError("--parallel requires --gpu to be set")
    if args.wds is False and len(args.data_path) == 0:
        raise cli.ValidationError("Must provide at least one data source: DATA_PATH positional argument or --wds")

    if args.wds is True:
        if args.wds_info is None and len(args.data_path) == 0:
            raise cli.ValidationError("--wds requires a data path unless --wds-info is provided")
        if len(args.data_path) > 1:
            raise cli.ValidationError(
                f"--wds can have at most 1 DATA_PATH positional argument, got {len(args.data_path)}"
            )
        if args.wds_info is None and len(args.data_path) == 1:
            data_path = args.data_path[0]
            if "://" in data_path and args.wds_size is None:
                raise cli.ValidationError("--wds-size is required for remote DATA_PATH")

    args.size = cli.parse_size(args.size)


def main(args: argparse.Namespace) -> None:
    validate_args(args)

    if settings.RESULTS_DIR.joinpath(args.dir).exists() is False:
        logger.info(f"Creating {settings.RESULTS_DIR.joinpath(args.dir)} directory...")
        settings.RESULTS_DIR.joinpath(args.dir).mkdir(parents=True)

    evaluate(args)
