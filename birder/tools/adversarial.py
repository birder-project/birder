import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from birder.adversarial.fgsm import FGSM
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.transforms.classification import inference_preset


def show_fgsm(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (net, class_to_idx, signature, rgb_values) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        inference=False,
        pts=False,
    )
    label_names = list(class_to_idx.keys())
    size = lib.get_size_from_signature(signature)[0]
    transform = inference_preset((size, size), rgb_values, 1.0)

    img: Image.Image = Image.open(args.image)
    input_tensor = transform(img).unsqueeze(dim=0).to(device)

    fgsm = FGSM(net, eps=args.eps)
    if args.target is not None:
        target = torch.tensor(class_to_idx[args.target]).unsqueeze(dim=0).to(device)
    else:
        target = None

    img = img.resize((size, size))
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
            "--epoch 0 --target Bluethroat --image 'data/training/Mallard/000117.jpeg'\n"
            "python -m birder.tools adversarial --method fgsm --network efficientnet_v2_m "
            "--epoch 0 --eps 0.02 --target Mallard --image 'data/validation/White-tailed eagle/000006.jpeg'\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--method", type=str, choices=["fgsm"], help="introspection method")
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required for most networks"
    )
    subparser.add_argument("-e", "--epoch", type=int, help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument("--eps", type=float, default=0.007, help="fgsm epsilon")
    subparser.add_argument("--target", type=str, help="target class, leave empty to use predicted class")
    subparser.add_argument("--image", type=str, required=True, help="input image")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.method == "fgsm":
        show_fgsm(args)
