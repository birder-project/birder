import argparse
import logging
from collections.abc import Callable
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from birder.adversarial.base import Attack
from birder.adversarial.base import AttackResult
from birder.adversarial.deepfool import DeepFool
from birder.adversarial.fgsm import FGSM
from birder.adversarial.pgd import PGD
from birder.adversarial.simba import SimBA
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import reverse_preset

logger = logging.getLogger(__name__)


def _load_model_and_transform(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, dict[str, int], RGBType, Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    (net, model_info) = fs_ops.load_model(
        device, args.network, tag=args.tag, epoch=args.epoch, inference=True, reparameterized=args.reparameterized
    )

    class_to_idx = model_info.class_to_idx
    rgb_stats = model_info.rgb_stats

    size = lib.get_size_from_signature(model_info.signature)
    transform = inference_preset(size, rgb_stats, 1.0)
    reverse_transform = reverse_preset(rgb_stats)

    return (net, class_to_idx, rgb_stats, transform, reverse_transform)


def _resolve_target(
    target_name: Optional[str], class_to_idx: dict[str, int], device: torch.device
) -> Optional[torch.Tensor]:
    if target_name is None:
        return None
    if target_name not in class_to_idx:
        raise ValueError(f"Unknown target class '{target_name}'")

    return torch.tensor([class_to_idx[target_name]], device=device, dtype=torch.long)


def _build_attack(args: argparse.Namespace, net: torch.nn.Module, rgb_stats: RGBType) -> Attack:
    if args.method == "fgsm":
        return FGSM(net, eps=args.eps, rgb_stats=rgb_stats)
    if args.method == "pgd":
        return PGD(
            net,
            eps=args.eps,
            steps=args.steps,
            step_size=args.step_size,
            random_start=args.random_start,
            rgb_stats=rgb_stats,
        )
    if args.method == "deepfool":
        return DeepFool(
            net,
            num_classes=args.deepfool_num_classes,
            overshoot=args.deepfool_overshoot,
            max_iter=args.deepfool_max_iter,
            rgb_stats=rgb_stats,
        )
    if args.method == "simba":
        return SimBA(
            net,
            step_size=args.step_size if args.step_size is not None else args.eps,
            max_iter=args.steps,
            rgb_stats=rgb_stats,
        )

    raise ValueError(f"Unsupported attack method '{args.method}'")


def _tensor_to_image(tensor: torch.Tensor, reverse_transform: Callable[..., torch.Tensor]) -> npt.NDArray[np.uint8]:
    img_tensor = reverse_transform(tensor).cpu()
    img = img_tensor.numpy()
    return np.moveaxis(img, 0, 2)


def _get_prediction(logits: torch.Tensor, label_names: list[str]) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    idx = int(np.argmax(probs))
    return (label_names[idx], float(probs[idx]))


def _display_results(
    original_img: npt.NDArray[np.uint8],
    adv_img: npt.NDArray[np.uint8],
    original_pred: tuple[str, float],
    adv_pred: tuple[str, float],
    success: Optional[bool],
    result: AttackResult,
) -> None:
    (orig_label, orig_prob) = original_pred
    (adv_label, adv_prob) = adv_pred

    # Log results
    logger.info(f"Original: {orig_label} ({orig_prob * 100:.2f}%)")
    logger.info(f"Adversarial: {adv_label} ({adv_prob * 100:.2f}%)")
    if success is not None:
        logger.info(f"Attack success: {success}")
    if result.num_queries is not None:
        logger.info(f"Model queries: {result.num_queries}")

    # Display images
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(original_img)
    ax1.set_title(f"{orig_label} {100 * orig_prob:.2f}%")
    ax1.axis("off")
    ax2.imshow(adv_img)
    ax2.set_title(f"{adv_label} {100 * adv_prob:.2f}%")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


def run_attack(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    (net, class_to_idx, rgb_stats, transform, reverse_transform) = _load_model_and_transform(args, device)
    label_names = [name for name, _idx in sorted(class_to_idx.items(), key=lambda item: item[1])]
    img = Image.open(args.image_path)
    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    target = _resolve_target(args.target, class_to_idx, device)
    attack = _build_attack(args, net, rgb_stats)
    result = attack(input_tensor, target=target)

    original_img = _tensor_to_image(input_tensor.squeeze(0).cpu(), reverse_transform)
    adv_img = _tensor_to_image(result.adv_inputs.squeeze(0).cpu(), reverse_transform)
    original_logits = result.logits
    if original_logits is None:
        with torch.no_grad():
            original_logits = net(input_tensor)

    original_pred = _get_prediction(original_logits, label_names)
    adv_pred = _get_prediction(result.adv_logits, label_names)
    success = bool(result.success.item()) if result.success is not None else None

    _display_results(original_img, adv_img, original_pred, adv_pred, success, result)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "adversarial",
        allow_abbrev=False,
        help="generate and visualize adversarial examples",
        description="generate and visualize adversarial examples",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools adversarial -n resnet_v2_50 -e 0 --method fgsm --eps 0.02 "
            "data/validation/Mallard/000112.jpeg\n"
            "python -m birder.tools adversarial -n efficientnet_v2_m -e 0 --method pgd --eps 0.02 --steps 10 "
            "data/validation/Mallard/000002.jpeg\n"
            "python -m birder.tools adversarial -n convnext_v2_tiny -e 0 --method deepfool "
            "data/validation/Bluethroat/000013.jpeg\n"
            "python -m birder.tools adversarial -n convnext_v2_tiny -e 0 --method simba --steps 1000 --step-size 0.1 "
            "data/validation/Bluethroat/000043.jpeg\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("-n", "--network", type=str, required=True, help="neural network to attack")
    subparser.add_argument("-t", "--tag", type=str, help="model tag")
    subparser.add_argument("-e", "--epoch", type=int, required=True, help="model checkpoint epoch")
    subparser.add_argument("--reparameterized", default=False, action="store_true", help="load reparameterized model")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use GPU")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="GPU device ID")
    subparser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["fgsm", "pgd", "deepfool", "simba"],
        help="adversarial attack method",
    )
    subparser.add_argument("--eps", type=float, default=0.007, help="perturbation budget in pixel space [0, 1]")
    subparser.add_argument("--target", type=str, help="target class name for targeted attack (omit for untargeted)")
    subparser.add_argument("--steps", type=int, default=10, help="number of iterations for iterative attacks")
    subparser.add_argument("--step-size", type=float, help="step size in pixel space (defaults to eps/steps for PGD)")
    subparser.add_argument(
        "--random-start", default=False, action="store_true", help="use random initialization for PGD"
    )
    subparser.add_argument(
        "--deepfool-num-classes", type=int, default=10, help="number of top classes to consider for DeepFool"
    )
    subparser.add_argument("--deepfool-overshoot", type=float, default=0.02, help="overshoot parameter for DeepFool")
    subparser.add_argument("--deepfool-max-iter", type=int, default=50, help="max iterations for DeepFool")
    subparser.add_argument("image_path", type=str, help="path to input image")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    run_attack(args)
