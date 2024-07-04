import argparse
import logging
from typing import Any

import torch

from birder.common import cli
from birder.conf import settings


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "fetch-model",
        help="download pretrained model",
        description="download pretrained model",
        epilog=(
            "Usage examples:\n"
            "python3 tool.py fetch-model mobilenet_v3_1_0.pt\n"
            "python3 tool.py fetch-model convnext_v2_4_0.pt\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("model_name", help="model name")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    dst = settings.MODELS_DIR.joinpath(args.model_name)
    if dst.exists() is True:
        logging.warning(f"Model {args.model_name} already exists... aborting")
        return

    url = f"https://f000.backblazeb2.com/file/birder/models/{args.model_name}"
    torch.hub.download_url_to_file(url, settings.MODELS_DIR.joinpath(args.model_name), progress=True)
