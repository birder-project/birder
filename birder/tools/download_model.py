import argparse
import logging
from typing import Any

from birder.common import cli
from birder.common.lib import get_pretrained_model_url
from birder.conf import settings
from birder.model_registry import registry

logger = logging.getLogger(__name__)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "download-model",
        allow_abbrev=False,
        help="download pretrained model",
        description="download pretrained model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools download-model mobilenet_v3_large_1\n"
            "python -m birder.tools download-model convnext_v2_tiny_0 --force\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--format",
        type=str,
        choices=["pt", "pt2", "pts", "ptl", "safetensors"],
        default="pt",
        help="model serialization format",
    )
    subparser.add_argument("--no-verify", action="store_true", help="skip SHA256 checksum verification")
    subparser.add_argument("--force", action="store_true", help="force download even if model already exists")
    subparser.add_argument("model_name", help="the model to download")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert (
        args.model_name in registry.list_pretrained_models()
    ), "Unknown model, see list-models tool for available options"

    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    model_metadata = registry.get_pretrained_metadata(args.model_name)
    if args.format not in model_metadata["formats"]:
        logger.warning(f"Available formats for {args.model_name} are: {list(model_metadata['formats'].keys())}")
        raise SystemExit(1)

    (model_file, url) = get_pretrained_model_url(args.model_name, args.format)
    dst = settings.MODELS_DIR.joinpath(model_file)
    if dst.exists() is True and args.force is False:
        logger.warning(f"File {model_file} already exists... aborting")
        raise SystemExit(1)

    if args.no_verify is True:
        checksum = None
    else:
        checksum = model_metadata["formats"][args.format]["sha256"]

    cli.download_file(url, dst, checksum, override=args.force)
