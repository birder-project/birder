import argparse
import logging
from typing import Any

from birder.common import cli
from birder.conf import settings
from birder.model_registry import registry


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "fetch-model",
        allow_abbrev=False,
        help="download pretrained model",
        description="download pretrained model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools fetch-model mobilenet_v3_large_1_0\n"
            "python -m birder.tools fetch-model convnext_v2_4_0 --force\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--format", type=str, choices=["pt", "pt2", "pts", "ptl"], default="pt", help="model serialization format"
    )
    subparser.add_argument("--force", action="store_true", help="force download even if model already exists")
    subparser.add_argument("model_name", choices=registry.list_pretrained_models(), help="the model to download")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if settings.MODELS_DIR.exists() is False:
        logging.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    model_info = registry.get_pretrained_info(args.model_name)
    if args.format not in model_info["formats"]:
        logging.warning(f"Available formats for {args.model_name} are: {list(model_info['formats'].keys())}")
        raise SystemExit(1)

    model_file = f"{args.model_name}.{args.format}"
    dst = settings.MODELS_DIR.joinpath(model_file)
    if dst.exists() is True and args.force is False:
        logging.warning(f"File {model_file} already exists... aborting")
        raise SystemExit(1)

    url = f"{settings.REGISTRY_BASE_UTL}/{model_file}"
    cli.download_file(url, dst, model_info["formats"][args.format]["sha256"], override=args.force)
