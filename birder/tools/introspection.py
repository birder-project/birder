import argparse
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.introspection import attention_rollout
from birder.introspection import gradcam
from birder.introspection import guided_backprop
from birder.net.base import BaseNet
from birder.transforms.classification import inference_preset


def _swin_reshape_transform(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(3))

    # Bring the channels to the first dimension like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)

    return result


def _deprocess_image(img: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """
    See https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """

    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)

    return np.array(img * 255).astype(np.uint8)


def show_attn_rollout(
    args: argparse.Namespace,
    net: BaseNet,
    _class_to_idx: dict[str, int],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
) -> None:
    img = Image.open(args.image_path)
    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    ar_net = attention_rollout.AttentionRollout(net, args.attn_layer_name)
    attn_rollout = ar_net(input_tensor, args.discard_ratio, args.head_fusion)
    mask = Image.fromarray(attn_rollout.numpy())

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    ax1.imshow(img.resize((args.size, args.size)))
    ax2.matshow(np.array(mask.resize((args.size, args.size))))
    ax3.matshow(np.array(mask.resize((args.size, args.size))))
    ax3.imshow(img.resize((args.size, args.size)), alpha=0.4)
    plt.show()


def show_guided_backprop(
    args: argparse.Namespace,
    net: BaseNet,
    class_to_idx: dict[str, int],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
) -> None:
    img = Image.open(args.image_path)
    rgb_img = np.array(img.resize((args.size, args.size))).astype(np.float32) / 255.0

    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    if args.target is not None:
        target = class_to_idx[args.target]
    else:
        target = None

    guided_bp = guided_backprop.GuidedBackpropReLUModel(net)
    bp_img = guided_bp(input_tensor, target_category=target)
    bp_img = _deprocess_image(bp_img * rgb_img)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(bp_img)
    ax2.imshow(rgb_img)
    plt.show()


def show_grad_cam(
    args: argparse.Namespace,
    net: BaseNet,
    class_to_idx: dict[str, int],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
) -> None:
    if args.network.startswith("swin_") is True:
        if args.reshape_size is None:
            args.reshape_size = int(args.size / (2**5))

        reshape_transform = partial(_swin_reshape_transform, height=args.reshape_size, width=args.reshape_size)

    else:
        reshape_transform = None

    img = Image.open(args.image_path)
    rgb_img = np.array(img.resize((args.size, args.size))).astype(np.float32) / 255.0

    block = getattr(net, args.block_name)
    target_layer = block[args.layer_num]
    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    grad_cam = gradcam.GradCAM(net, target_layer, reshape_transform=reshape_transform)
    if args.target is not None:
        target = gradcam.ClassifierOutputTarget(class_to_idx[args.target])

    else:
        target = None

    grayscale_cam = grad_cam(input_tensor, target=target)
    grayscale_cam = grayscale_cam[0, :]
    visualization = gradcam.show_cam_on_image(rgb_img, grayscale_cam)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(visualization)
    ax2.imshow(rgb_img)
    plt.show()


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
            "--reshape-size 20 data/training/Fieldfare/000002.jpeg\n"
            "python -m birder.tools introspection --method attn-rollout -n vitreg4_b16 -t mim -e 100 "
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
    subparser.add_argument("--size", type=int, help="image size for inference (defaults to model signature)")
    subparser.add_argument(
        "--target", type=str, help="target class, leave empty to use predicted class (gradcam and guided-backprop only)"
    )
    subparser.add_argument("--block-name", type=str, default="body", help="target block (gradcam only)")
    subparser.add_argument(
        "--layer-num", type=int, default=-1, help="target layer, index for target block (gradcam only)"
    )
    subparser.add_argument("--reshape-size", type=int, help="2d projection for transformer models (gradcam only)")
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
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logging.info(f"Using device {device}")

    (net, class_to_idx, signature, rgb_stats) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        inference=False,
        reparameterized=args.reparameterized,
    )
    if args.size is None:
        args.size = lib.get_size_from_signature(signature)[0]

    else:
        net.adjust_size(args.size)

    transform = inference_preset((args.size, args.size), rgb_stats, 1.0)

    if args.method == "gradcam":
        show_grad_cam(args, net, class_to_idx, transform, device)
    elif args.method == "guided-backprop":
        show_guided_backprop(args, net, class_to_idx, transform, device)
    elif args.method == "attn-rollout":
        show_attn_rollout(args, net, class_to_idx, transform, device)
