import argparse
from typing import Any

from rich.columns import Columns
from rich.console import Console

from birder.common import cli
from birder.model_registry import Task
from birder.model_registry import registry


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "list-models",
        help="list available models",
        description="list available models",
        epilog=(
            "Usage examples:\n"
            "python tool.py list-models\n"
            "python tool.py list-models --detection\n"
            "python tool.py list-models --pretrained\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--detection", default=False, action="store_true", help="list detection models")
    subparser.add_argument("--pretrained", default=False, action="store_true", help="list pretrained models")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:

    if args.detection is True:
        model_list = registry.list_models(task=Task.OBJECT_DETECTION)

    elif args.pretrained is True:
        model_list = registry.list_pretrained_models()

    else:
        model_list = registry.list_models(task=Task.IMAGE_CLASSIFICATION)

    console = Console()
    console.print(Columns(model_list, padding=(0, 3), equal=True, column_first=True, title="[bold]Models[/bold]"))
