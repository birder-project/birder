import argparse
import json
import logging
import time
from typing import Any

import torch
import torch.ao.quantization
from torch.ao.quantization.quantize_fx import convert_fx
from torch.ao.quantization.quantize_fx import prepare_fx
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.transforms.classification import inference_preset

logger = logging.getLogger(__name__)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "quantize-model",
        allow_abbrev=False,
        help="quantize model",
        description="quantize model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools quantize-model --network shufflenet_v2 --net-param 2 --epoch 0\n"
            "python -m birder.tools quantize-model -n convnext_v2_tiny -e 0 --qbackend x86 \n"
            "python -m birder.tools quantize-model --network densenet_121 -e 100 --num-calibration-batches 256\n"
            "python -m birder.tools quantize-model -n efficientnet_v2_s -e 200 --qbackend x86\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to load (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required by some networks"
    )
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument("-f", "--force", action="store_true", help="override existing model")
    subparser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=8,
        help="number of preprocessing workers",
    )
    subparser.add_argument(
        "--qbackend",
        type=str,
        choices=["qnnpack", "x86", "onednn"],
        default="qnnpack",
        help="quantized backend",
    )
    subparser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    subparser.add_argument(
        "--num-calibration-batches",
        default=128,
        type=int,
        help="number of batches of training set for observer calibration",
    )
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="training directory path"
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    network_name = get_network_name(args.network, net_param=args.net_param, tag=args.tag)
    model_path = fs_ops.model_path(network_name, epoch=args.epoch, quantized=True, pts=True)
    if model_path.exists() is True and args.force is False:
        logger.warning("Quantized model already exists... aborting")
        raise SystemExit(1)

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        inference=True,
        reparameterized=args.reparameterized,
    )
    net.eval()
    task = net.task
    size = lib.get_size_from_signature(signature)

    # Set calibration data
    full_dataset = ImageFolder(args.data_path, transform=inference_preset(size, rgb_stats, 1.0))
    calibration_dataset = Subset(full_dataset, indices=list(range(args.batch_size * args.num_calibration_batches)))
    calibration_data_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Quantization
    tic = time.time()
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(args.qbackend)

    example_inputs = next(iter(calibration_data_loader))[0]
    prepared_net = prepare_fx(net, qconfig_mapping, example_inputs)

    with tqdm(total=len(calibration_dataset), initial=0, unit="images", unit_scale=True, leave=False) as progress:
        with torch.inference_mode():
            for inputs, targets in calibration_data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                _ = prepared_net(inputs)

                # Update progress bar
                progress.update(n=args.batch_size)

    net = convert_fx(prepared_net)

    toc = time.time()
    (minutes, seconds) = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to quantize model")

    model_path = fs_ops.model_path(network_name, epoch=args.epoch, quantized=True, pts=True)
    logger.info(f"Saving quantized TorchScript model {model_path}...")

    # Convert to TorchScript
    scripted_module = torch.jit.script(net)
    fs_ops.save_pts(scripted_module, model_path, task, class_to_idx, signature, rgb_stats)

    if args.qbackend == "qnnpack":
        model_path = fs_ops.model_path(network_name, epoch=args.epoch, quantized=True, lite=True)
        if model_path.exists() is True and args.force is False:
            logger.warning("Quantized model lite already exists... aborting")
            raise SystemExit(1)

        logger.info(f"Saving quantized TorchScript model {model_path}...")
        optimized_scripted_module = optimize_for_mobile(scripted_module)
        optimized_scripted_module._save_for_lite_interpreter(  # pylint: disable=protected-access
            str(model_path),
            _extra_files={
                "task": task,
                "class_to_idx": json.dumps(class_to_idx),
                "signature": json.dumps(signature),
                "rgb_stats": json.dumps(rgb_stats),
            },
        )
