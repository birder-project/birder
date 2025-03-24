import argparse
from typing import Any

import torch
from rich.columns import Columns
from rich.console import Console

from birder.common import cli
from birder.common import fs_ops
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionSignatureType


def get_model_info(net: torch.nn.Module) -> dict[str, float]:
    num_params = 0
    param_size = 0
    buffer_size = 0
    for param in net.parameters():
        num_params += param.numel()
        param_size += param.numel() * param.element_size()

    for buffer in net.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    return {"num_params": num_params, "model_size": param_size + buffer_size}


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "model-info",
        allow_abbrev=False,
        help="print information about the model",
        description="print information about the model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools model-info -n deit_b16 -t intermediate -e 0\n"
            "python -m birder.tools model-info --network squeezenet --epoch 100\n"
            "python -m birder.tools model-info --network densenet_121 -e 100 --pt2\n"
            "python -m birder.tools model-info -n efficientnet_v2_m -e 200 --lite\n"
            "python -m birder.tools model-info --network faster_rcnn --backbone resnext_101 -e 0\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to load (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required by some networks"
    )
    subparser.add_argument(
        "--backbone",
        type=str,
        choices=registry.list_models(net_type=DetectorBackbone),
        help="the neural network to used as backbone",
    )
    subparser.add_argument(
        "--backbone-param",
        type=float,
        help="network specific parameter, required by some networks (for the backbone)",
    )
    subparser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument("--pts", default=False, action="store_true", help="load torchscript network")
    subparser.add_argument("--pt2", default=False, action="store_true", help="load standardized model")
    subparser.add_argument("--st", "--safetensors", default=False, action="store_true", help="load Safetensors weights")
    subparser.add_argument("--classes", default=False, action="store_true", help="print all classes")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    # Load model
    device = torch.device("cpu")
    signature: SignatureType | DetectionSignatureType
    backbone_custom_config = None
    if args.backbone is None:
        (net, (class_to_idx, signature, rgb_stats, custom_config)) = fs_ops.load_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            epoch=args.epoch,
            inference=True,
            reparameterized=args.reparameterized,
            pts=args.pts,
            pt2=args.pt2,
            st=args.st,
        )

    else:
        (net, (class_to_idx, signature, rgb_stats, custom_config, backbone_custom_config)) = (
            fs_ops.load_detection_model(
                device,
                args.network,
                net_param=args.net_param,
                tag=args.tag,
                backbone=args.backbone,
                backbone_param=args.backbone_param,
                backbone_tag=args.backbone_tag,
                epoch=args.epoch,
                inference=True,
                pts=args.pts,
                pt2=args.pt2,
                st=args.st,
            )
        )

    model_info = get_model_info(net)
    is_nan = torch.stack([torch.isnan(p).any() for p in net.parameters()]).any().item()

    console = Console()
    console.print(f"Network type: [bold]{type(net).__name__}[/bold], with task={net.task}")
    console.print(f"Network signature: {signature}")
    console.print(f"Network rgb values: {rgb_stats}")
    if custom_config is not None:
        console.print(f"Network has saved custom config: {custom_config}")
    if backbone_custom_config is not None:
        console.print(f"Network backbone has saved custom config: {backbone_custom_config}")

    console.print(f"Number of parameters: {model_info['num_params']:,}")
    console.print(f"Model size (inc. buffers): {(model_info['model_size']) / 1024**2:,.2f} [bold]MB[/bold]")
    console.print()
    if args.classes is True:
        console.print(Columns(list(class_to_idx.keys()), column_first=True, title="[bold]Class list[/bold]"))
    if is_nan is True:
        console.print()
        console.print("[red]Warning, NaN detected at the model weights[/red]")
