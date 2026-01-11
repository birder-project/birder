import argparse
import itertools
import logging
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.data.transforms.classification import inference_preset

try:
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config

    _HAS_TORCHAO = True
except ImportError:
    _HAS_TORCHAO = False

try:
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import get_symmetric_quantization_config

    _HAS_EXECUTORCH = True
except ImportError:
    _HAS_EXECUTORCH = False

logger = logging.getLogger(__name__)


def _build_quantizer(backend: str) -> Any:
    assert _HAS_TORCHAO, "'pip install torchao' to use quantization"
    if backend == "xnnpack":
        assert _HAS_EXECUTORCH, "'pip install executorch' to use quantization"
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config())
        return quantizer

    if backend == "x86":
        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config())
        return quantizer

    raise ValueError(f"Unsupported backend: {backend}")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "quantize-model",
        allow_abbrev=False,
        help="quantize model",
        description="quantize model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools quantize-model -n convnext_v2_tiny -t eu-common\n"
            "python -m birder.tools quantize-model --network densenet_121 -e 100 --num-calibration-batches 256\n"
            "python -m birder.tools quantize-model -n efficientnet_v2_s -e 200 --qbackend xnnpack --batch-size 1\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to load (i.e. resnet_v2_50)"
    )
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from the training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument("-f", "--force", action="store_true", help="override existing model")
    subparser.add_argument("-j", "--num-workers", type=int, default=4, help="number of preprocessing workers")
    subparser.add_argument(
        "--qbackend", type=str, choices=["x86", "xnnpack"], default="x86", help="quantization backend"
    )
    subparser.add_argument("--batch-size", type=int, default=1, metavar="N", help="the batch size")
    subparser.add_argument(
        "--num-calibration-batches",
        default=256,
        type=int,
        help="number of batches of training set for observer calibration",
    )
    subparser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="training directory path"
    )
    subparser.set_defaults(func=main)


# pylint: disable=too-many-locals
def main(args: argparse.Namespace) -> None:
    network_name = get_network_name(args.network, tag=args.tag)
    model_path = fs_ops.model_path(network_name, epoch=args.epoch, quantized=True, pt2=True)
    if model_path.exists() is True and args.force is False:
        logger.warning("Quantized model already exists... aborting")
        raise SystemExit(1)

    device = torch.device("cpu")

    # Load model
    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_model(
        device, args.network, tag=args.tag, epoch=args.epoch, inference=True, reparameterized=args.reparameterized
    )
    net.eval()
    task = net.task
    size = lib.get_size_from_signature(signature)

    # Set calibration data
    full_dataset = ImageFolder(args.data_path, transform=inference_preset(size, rgb_stats, 1.0))
    num_calibration_samples = min(
        len(full_dataset),
        args.batch_size * args.num_calibration_batches,
    )

    indices = torch.randperm(len(full_dataset))[:num_calibration_samples].tolist()
    calibration_dataset = Subset(full_dataset, indices=indices)
    calibration_data_loader = DataLoader(
        calibration_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Quantization
    tic = time.time()
    quantizer = _build_quantizer(args.qbackend)
    calibration_iter = iter(calibration_data_loader)

    first_batch = next(calibration_iter)
    example_inputs = (first_batch[0].to(device),)

    # batch_dim = torch.export.Dim("batch", min=1, max=4096)
    # dynamic_shapes = ({0: batch_dim},)

    with torch.no_grad():
        exported_net = torch.export.export(net, example_inputs).module()
        prepared_net = prepare_pt2e(exported_net, quantizer)

    with tqdm(total=num_calibration_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        with torch.inference_mode():
            for inputs, _ in itertools.chain([first_batch], calibration_iter):
                inputs = inputs.to(device)
                prepared_net(inputs)

                # Update progress bar
                progress.update(n=inputs.shape[0])

    with torch.no_grad():
        quantized_net = convert_pt2e(prepared_net)
        exported_quantized_net = torch.export.export(quantized_net, example_inputs)

    toc = time.time()
    (minutes, seconds) = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to quantize model")

    model_path = fs_ops.model_path(network_name, epoch=args.epoch, quantized=True, pt2=True)
    logger.info(f"Saving quantized PT2 model {model_path}...")
    fs_ops.save_pt2(exported_quantized_net, model_path, task, class_to_idx, signature, rgb_stats)
