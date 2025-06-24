import argparse
import logging
from collections.abc import Callable
from typing import Any

import torch

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.data.transforms.classification import inference_preset
from birder.introspection import AttentionRolloutInterpreter
from birder.introspection import GradCamInterpreter
from birder.introspection import GuidedBackpropInterpreter
from birder.net.base import BaseNet

logger = logging.getLogger(__name__)


def _nhwc_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 3, 1, 2).contiguous()


def show_attn_rollout(
    args: argparse.Namespace,
    net: BaseNet,
    _class_to_idx: dict[str, int],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
) -> None:
    ar = AttentionRolloutInterpreter(net, device, transform, args.attn_layer_name, args.discard_ratio, args.head_fusion)
    result = ar.interpret(args.image_path)
    result.show()


def show_guided_backprop(
    args: argparse.Namespace,
    net: BaseNet,
    class_to_idx: dict[str, int],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
) -> None:
    if args.target is not None:
        target = class_to_idx[args.target]
    else:
        target = None

    guided_bp = GuidedBackpropInterpreter(net, device, transform)
    result = guided_bp.interpret(args.image_path, target_class=target)
    result.show()


def show_grad_cam(
    args: argparse.Namespace,
    net: BaseNet,
    class_to_idx: dict[str, int],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
) -> None:
    if args.channels_last is True:
        reshape_transform = _nhwc_reshape_transform
    else:
        reshape_transform = None

    block = getattr(net, args.block_name)
    target_layer = block[args.layer_num]

    if args.target is not None:
        target = class_to_idx[args.target]
    else:
        target = None

    grad_cam = GradCamInterpreter(net, device, transform, target_layer, reshape_transform=reshape_transform)
    result = grad_cam.interpret(args.image_path, target_class=target)
    result.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "introspection",
        allow_abbrev=False,
        help="computer vision introspection and explainability",
        description="computer vision introspection and explainability",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools introspection --method gradcam --network efficientnet_v2_m "
            "--epoch 200 'data/training/European goldfinch/000300.jpeg'\n"
            "python -m birder.tools introspection --method gradcam -n resnest_50 --epoch 300 "
            "data/index5.jpeg --target 'Grey heron'\n"
            "python -m birder.tools introspection --method guided-backprop -n efficientnet_v2_s "
            "-e 0 'data/training/European goldfinch/000300.jpeg'\n"
            "python -m birder.tools introspection --method gradcam -n swin_transformer_v1_b -e 85 --layer-num -4 "
            "--channels-last data/training/Fieldfare/000002.jpeg\n"
            "python -m birder.tools introspection --method attn-rollout -n vit_reg4_b16 -t mim -e 100 "
            " data/validation/Bluethroat/000013.jpeg\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--method", type=str, choices=["gradcam", "guided-backprop", "attn-rollout"], help="introspection method"
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required by some networks"
    )
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    subparser.add_argument(
        "--target", type=str, help="target class, leave empty to use predicted class (gradcam and guided-backprop only)"
    )
    subparser.add_argument("--block-name", type=str, default="body", help="target block (gradcam only)")
    subparser.add_argument(
        "--layer-num", type=int, default=-1, help="target layer, index for target block (gradcam only)"
    )
    subparser.add_argument(
        "--channels-last", default=False, action="store_true", help="channels last model, like swin (gradcam only)"
    )
    subparser.add_argument(
        "--attn-layer-name", type=str, default="self_attention", help="attention layer name (attn-rollout only)"
    )
    subparser.add_argument(
        "--head-fusion",
        type=str,
        choices=["mean", "max", "min"],
        default="max",
        help="how to fuse the attention heads for attention rollout (attn-rollout only)",
    )
    subparser.add_argument(
        "--discard-ratio",
        type=float,
        default=0.9,
        help="how many of the lowest attention paths should be discarded (attn-rollout only)",
    )
    subparser.add_argument("image_path", type=str, help="input image path")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    (net, model_info) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        new_size=args.size,
        inference=False,
        reparameterized=args.reparameterized,
    )
    if args.size is None:
        args.size = lib.get_size_from_signature(model_info.signature)

    transform = inference_preset(args.size, model_info.rgb_stats, 1.0)

    if args.method == "gradcam":
        show_grad_cam(args, net, model_info.class_to_idx, transform, device)
    elif args.method == "guided-backprop":
        show_guided_backprop(args, net, model_info.class_to_idx, transform, device)
    elif args.method == "attn-rollout":
        show_attn_rollout(args, net, model_info.class_to_idx, transform, device)
