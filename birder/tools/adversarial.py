import argparse
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from birder.adversarial.fgsm import FGSM
from birder.adversarial.pgd import PGD
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import reverse_preset

logger = logging.getLogger(__name__)


def show_pgd(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        inference=True,
        reparameterized=args.reparameterized,
    )
    label_names = list(class_to_idx.keys())
    size = lib.get_size_from_signature(signature)
    transform = inference_preset(size, rgb_stats, 1.0)
    reverse_transform = reverse_preset(rgb_stats)

    img: Image.Image = Image.open(args.image_path)
    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    pgd = PGD(net, eps=args.eps, max_delta=0.012, steps=10, random_start=True)
    if args.target is not None:
        target = torch.tensor(class_to_idx[args.target]).unsqueeze(dim=0).to(device)
    else:
        target = None

    img = img.resize(size)
    pgd_response = pgd(input_tensor, target=target)
    perturbation = reverse_transform(pgd_response.adv_img).cpu().detach().numpy().squeeze()
    pgd_img = np.moveaxis(perturbation, 0, 2)

    # Get predictions and probabilities
    prob = pgd_response.out.cpu().detach().numpy().squeeze()
    adv_prob = pgd_response.adv_out.cpu().detach().numpy().squeeze()
    idx = np.argmax(prob)
    adv_idx = np.argmax(adv_prob)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(img)
    ax1.set_title(f"{label_names[idx]} {100 * prob[idx]:.2f}%")
    ax2.imshow(pgd_img)
    ax2.set_title(f"{label_names[adv_idx]} {100 * adv_prob[adv_idx]:.2f}%")
    plt.show()


def show_fgsm(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        inference=False,
        reparameterized=args.reparameterized,
    )
    label_names = list(class_to_idx.keys())
    size = lib.get_size_from_signature(signature)
    transform = inference_preset(size, rgb_stats, 1.0)

    img: Image.Image = Image.open(args.image_path)
    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    fgsm = FGSM(net, eps=args.eps)
    if args.target is not None:
        target = torch.tensor(class_to_idx[args.target]).unsqueeze(dim=0).to(device)
    else:
        target = None

    img = img.resize(size)
    fgsm_response = fgsm(input_tensor, target=target)
    perturbation = fgsm_response.perturbation.cpu().detach().numpy().squeeze()
    fgsm_img = (np.array(img).astype(np.float32) / 255.0) + np.moveaxis(perturbation, 0, 2)
    fgsm_img = np.clip(fgsm_img, 0, 1)

    # Get predictions and probabilities
    prob = fgsm_response.out.cpu().detach().numpy().squeeze()
    adv_prob = fgsm_response.adv_out.cpu().detach().numpy().squeeze()
    idx = np.argmax(prob)
    adv_idx = np.argmax(adv_prob)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(img)
    ax1.set_title(f"{label_names[idx]} {100 * prob[idx]:.2f}%")
    ax2.imshow(fgsm_img)
    ax2.set_title(f"{label_names[adv_idx]} {100 * adv_prob[adv_idx]:.2f}%")
    plt.show()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "adversarial",
        allow_abbrev=False,
        help="deep learning adversarial attacks",
        description="deep learning adversarial attacks",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools adversarial --method fgsm --network efficientnet_v2_s "
            "--epoch 0 --target Bluethroat 'data/training/Mallard/000117.jpeg'\n"
            "python -m birder.tools adversarial --method fgsm --network efficientnet_v2_m "
            "--epoch 0 --eps 0.02 --target Mallard 'data/validation/White-tailed eagle/000006.jpeg'\n"
            "python tool.py adversarial --method pgd --network caformer_s18 -e 0 "
            "data/validation/Arabian babbler/000001.jpeg\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--method", type=str, choices=["fgsm", "pgd"], help="introspection method")
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
    subparser.add_argument("--eps", type=float, default=0.007, help="fgsm epsilon")
    subparser.add_argument("--target", type=str, help="target class, leave empty to use predicted class")
    subparser.add_argument("image_path", type=str, help="input image path")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.method == "fgsm":
        show_fgsm(args)
    elif args.method == "pgd":
        show_pgd(args)
