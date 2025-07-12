import argparse
import fnmatch
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.model_registry import Task
from birder.model_registry import registry
from birder.model_registry.model_registry import group_sort
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "list-models",
        allow_abbrev=False,
        help="list available models",
        description="list available models",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools list-models\n"
            "python -m birder.tools list-models --classification\n"
            "python -m birder.tools list-models --classification --detector-backbone\n"
            "python -m birder.tools list-models --pretrain-encoder\n"
            "python -m birder.tools list-models --detection\n"
            "python -m birder.tools list-models --pretrained\n"
            "python -m birder.tools list-models --pretrained\n"
            "python -m birder.tools list-models --pretrained --detection --verbose\n"
            "python tool.py list-models --pretrained --verbose --filter '*mobile*'\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    task_group = subparser.add_mutually_exclusive_group(required=False)
    task_group.add_argument("--classification", default=False, action="store_true", help="list classification models")
    task_group.add_argument("--detection", default=False, action="store_true", help="list detection models")
    task_group.add_argument("--mim", default=False, action="store_true", help="list MIM models")
    task_group.add_argument("--ssl", default=False, action="store_true", help="list SSL models")

    subparser.add_argument("--pretrained", default=False, action="store_true", help="list pretrained models")

    type_group = subparser.add_argument_group()
    type_group.add_argument(
        "--detector-backbone", default=False, action="store_true", help="list detector backbone models"
    )
    type_group.add_argument(
        "--pretrain-encoder", default=False, action="store_true", help="list models that support mim pretraining"
    )
    type_group.add_argument(
        "--token-omission",
        default=False,
        action="store_true",
        help="list models supporting token omission for sparse processing",
    )
    type_group.add_argument(
        "--token-retention", default=False, action="store_true", help="list models supporting masked token retention"
    )

    subparser.add_argument("--filter", type=str, help="filter results with a fnmatch type filter)")
    subparser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="enable verbose output with additional model details",
    )
    subparser.set_defaults(func=main)


# pylint: disable=too-many-branches
def main(args: argparse.Namespace) -> None:
    types: list[type] = []
    if args.detector_backbone is True:
        types.append(DetectorBackbone)
    if args.pretrain_encoder is True:
        types.append(PreTrainEncoder)
    if args.token_omission is True:
        types.append(MaskedTokenOmissionMixin)
    if args.token_retention is True:
        types.append(MaskedTokenRetentionMixin)

    if len(types) == 0:
        t = None
    else:
        t = tuple(types)

    # Determine the task based on the selected flags
    task = None
    if args.classification is True:
        task = Task.IMAGE_CLASSIFICATION
    elif args.detection is True:
        task = Task.OBJECT_DETECTION
    elif args.mim is True:
        task = Task.MASKED_IMAGE_MODELING
    elif args.ssl is True:
        task = Task.SELF_SUPERVISED_LEARNING

    if args.pretrained is True:
        model_list = registry.list_pretrained_models(task=task)
    else:
        model_list = registry.list_models(task=task, net_type=t)

    model_list = group_sort(model_list)
    if args.filter is not None:
        model_list = fnmatch.filter(model_list, args.filter)

    console = Console()
    if args.verbose is True:
        if args.pretrained is True:
            table = Table(show_header=True, header_style="bold dark_magenta")
            table.add_column("Model name")
            table.add_column("Format", style="dim")
            table.add_column("File size", justify="right")
            table.add_column("Resolution", justify="right")
            table.add_column("Description")
            for model_name in model_list:
                model_metadata = registry.get_pretrained_metadata(model_name)
                for format_name, format_info in model_metadata["formats"].items():
                    table.add_row(
                        model_name,
                        format_name,
                        f"{format_info['file_size']}MB",
                        "x".join(str(x) for x in model_metadata["resolution"]),
                        model_metadata["description"],
                    )

            console.print(table)

        else:
            raise NotImplementedError
            # table = Table(show_header=True, header_style="bold dark_magenta")
            # table.add_column("Model name")
            # table.add_column("Description")
            # for model_name in model_list:
            #     if model_name in registry.aliases:
            #         desc = ""

            #     else:
            #         net = registry.all_nets[model_name]
            #         desc = sys.modules[net.__module__].__doc__
            #         if desc is not None:
            #             desc = desc.strip("\n")

            #     table.add_row(model_name, desc)

            # console.print(table)

    else:
        console.print(
            Columns(
                model_list,
                padding=(0, 3),
                equal=True,
                column_first=True,
                title=f"[bold]{len(model_list)} Models[/bold]",
            )
        )
