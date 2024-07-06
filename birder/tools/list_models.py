import argparse
from typing import Any

from rich.columns import Columns
from rich.console import Console

from birder.common import cli
from birder.conf.registry import MODEL_REGISTRY
from birder.core.net.base import _ALIAS
from birder.core.net.base import _BASE_NETWORKS


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "list-models",
        help="list available models",
        description="list available models",
        epilog=(
            "Usage examples:\n"
            "python tool.py list-models\n"
            "python tool.py list-models --aliases\n"
            "python tool.py list-models --registry\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--aliases", default=False, action="store_true", help="list model aliases")
    subparser.add_argument("--registry", default=False, action="store_true", help="list registry models")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:

    if args.aliases is True:
        model_list = _ALIAS

    elif args.registry is True:
        model_list = sorted(list(MODEL_REGISTRY.keys()))

    else:
        model_list = sorted(_BASE_NETWORKS)

    console = Console()
    console.print(Columns(model_list, padding=(0, 3), equal=True, column_first=True, title="[bold]Models[/bold]"))
