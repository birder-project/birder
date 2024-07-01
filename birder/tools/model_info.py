import argparse
from typing import Any
from typing import Union

import torch
from rich.columns import Columns
from rich.console import Console

from birder.common import cli
from birder.core.net.base import DetectorBackbone
from birder.core.net.base import SignatureType
from birder.core.net.base import network_names_filter
from birder.core.net.detection.base import DetectionSignatureType


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "model-info",
        help="print information about the model",
        description="print information about the model",
        epilog=(
            "Usage examples:\n"
            "python3 tool.py model-info -n deit -p 2 -t intermediate -e 0\n"
            "python3 tool.py model-info --network squeezenet --epoch 100\n"
            "python3 tool.py model-info --network densenet -p 121 -e 100 --pt2\n"
            "python3 tool.py model-info -n efficientnet_v2 -p 1 -e 200 --lite\n"
            "python3 tool.py model-info --network faster_rcnn --backbone resnext "
            "--backbone-param 101 -e 0\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to load (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required by most networks"
    )
    subparser.add_argument(
        "--backbone",
        type=str,
        choices=network_names_filter(DetectorBackbone),
        help="the neural network to used as backbone",
    )
    subparser.add_argument(
        "--backbone-param",
        type=float,
        help="network specific parameter, required by most networks (for the backbone)",
    )
    subparser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    subparser.add_argument("-e", "--epoch", type=int, default=0, help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    # Load model
    device = torch.device("cpu")
    signature: Union[SignatureType, DetectionSignatureType]
    if args.backbone is None:
        (net, class_to_idx, signature, rgb_values) = cli.load_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            epoch=args.epoch,
            inference=True,
            script=False,
        )

    else:
        (net, class_to_idx, signature, rgb_values) = cli.load_detection_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_tag=args.backbone_tag,
            epoch=args.epoch,
            inference=True,
            script=False,
        )

    console = Console()
    console.print(f"Network type: [bold]{type(net).__name__}[/bold], with task={net.task}")
    console.print(f"Network signature: {signature}")
    console.print(f"Network rgb values: {rgb_values}")
    console.print(f"Number of parameters: {sum(p.numel() for p in net.parameters()):,}")
    console.print()
    console.print(Columns(list(class_to_idx.keys()), column_first=True, title="[bold]Class list[/bold]"))
