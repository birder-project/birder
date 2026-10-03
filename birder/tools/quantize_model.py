import argparse
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import get_args

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common import training_utils
from birder.conf import settings
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import inference_preset
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionSignatureType
from birder.version import __version__

try:
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config

    _HAS_TORCHAO = True
except ImportError:
    _HAS_TORCHAO = False

try:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import get_symmetric_quantization_config
    from executorch.exir import to_edge_transform_and_lower

    _HAS_EXECUTORCH = True
except ImportError:
    _HAS_EXECUTORCH = False

logger = logging.getLogger(__name__)


def _build_quantizer(backend: str, per_channel: bool, dynamic_quantization: bool) -> Any:
    assert _HAS_TORCHAO, "'pip install torchao' to use quantization"
    if backend == "xnnpack":
        assert _HAS_EXECUTORCH, "'pip install executorch' to use quantization"
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(
                is_per_channel=per_channel,
                is_dynamic=dynamic_quantization,
            )
        )
        return quantizer

    if backend == "x86":
        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config(is_dynamic=dynamic_quantization))
        return quantizer

    raise ValueError(f"Unsupported backend: {backend}")


def _save_pte(
    exported_net: torch.export.ExportedProgram,
    model_path: str | Path,
    task: str,
    class_to_idx: dict[str, int],
    signature: SignatureType | DetectionSignatureType,
    rgb_stats: RGBType,
) -> None:
    metadata = {
        "birder_version": __version__,
        "task": task,
        "class_to_idx": class_to_idx,
        "signature": signature,
        "rgb_stats": rgb_stats,
    }
    edge_program = to_edge_transform_and_lower(
        exported_net,
        partitioner=[XnnpackPartitioner()],
        constant_methods={"get_metadata": json.dumps(metadata)},
    )
    executorch_program = edge_program.to_executorch()
    with open(model_path, "wb") as handle:
        handle.write(executorch_program.buffer)

    with open(f"{model_path}_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "quantize-model",
        allow_abbrev=False,
        help="quantize model",
        description="quantize model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools quantize-model -n convnext_v2_tiny -t eu-common --dynamic-size\n"
            "python -m birder.tools quantize-model --network densenet_121 -e 100 --num-calibration-batches 256\n"
            "python -m birder.tools quantize-model -n efficientnet_v2_s -e 200 --qbackend xnnpack --batch-size 1\n"
            "python -m birder.tools quantize-model -n hgnet_v2_b4 --qbackend xnnpack --pte\n"
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
    subparser.add_argument(
        "-j", "--num-workers", type=int, default=4, metavar="N", help="number of preprocessing workers"
    )
    subparser.add_argument(
        "--qbackend", type=str, choices=["x86", "xnnpack"], default="x86", help="quantization backend"
    )
    subparser.add_argument(
        "--per-channel",
        default=False,
        action="store_true",
        help="use a separate quantization scale per output channel (XNNPACK only)",
    )
    subparser.add_argument(
        "--dynamic-quantization", default=False, action="store_true", help="use dynamic quantization"
    )
    subparser.add_argument(
        "--pte", default=False, action="store_true", help="lower quantized model to ExecuTorch PTE format"
    )
    subparser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="calibration batch size (also the static export batch size)",
    )
    subparser.add_argument(
        "--num-calibration-batches",
        type=int,
        default=512,
        metavar="N",
        help="number of batches of training set for static observer calibration",
    )
    subparser.add_argument("--dynamic-batch", action="store_true", help="export with dynamic batch size")
    subparser.add_argument(
        "--max-batch-size", type=int, metavar="N", help="upper batch-size bound for --pte --dynamic-batch"
    )
    subparser.add_argument("--dynamic-size", default=False, action="store_true", help="export with dynamic input H/W")
    subparser.add_argument(
        "--trace-size",
        type=int,
        nargs="+",
        metavar=("H", "W"),
        help="sample H/W used for export tracing, does not resize the model",
    )
    subparser.add_argument(
        "--max-size", type=int, nargs="+", metavar=("H", "W"), help="upper H/W bounds for --pte --dynamic-size"
    )
    subparser.add_argument("--seed", type=int, help="set random seed for better reproducibility")
    subparser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="training directory path"
    )
    subparser.add_argument(
        "--img-loader",
        type=str,
        choices=get_args(ImageLoaderName),
        default="pil",
        help="backend to load and decode calibration images",
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    args.trace_size = cli.parse_size(args.trace_size)
    args.max_size = cli.parse_size(args.max_size)

    if args.pte is True and args.qbackend != "xnnpack":
        raise cli.ValidationError("--pte requires --qbackend xnnpack")
    if args.max_batch_size is not None and (args.pte is False or args.dynamic_batch is False):
        raise cli.ValidationError("--max-batch-size requires --pte --dynamic-batch")
    if args.pte is True and args.dynamic_batch is True:
        if args.max_batch_size is None:
            raise cli.ValidationError("--pte --dynamic-batch requires --max-batch-size")
    if args.max_size is not None and (args.pte is False or args.dynamic_size is False):
        raise cli.ValidationError("--max-size requires --pte --dynamic-size")
    if args.pte is True and args.dynamic_size is True and args.max_size is None:
        raise cli.ValidationError("--pte --dynamic-size requires --max-size")
    if args.trace_size is not None and args.dynamic_size is False:
        raise cli.ValidationError("--trace-size requires --dynamic-size")

    if args.seed is not None:
        training_utils.set_random_seeds(args.seed)

    network_name = lib.get_network_name(args.network, tag=args.tag)
    model_path = fs_ops.model_path(network_name, epoch=args.epoch, quantized=True, pt2=True)
    if args.pte is True:
        model_path = model_path.with_suffix(".pte")
    if model_path.exists() is True and args.force is False:
        logger.warning("Quantized model already exists... aborting")
        raise SystemExit(1)

    device = torch.device("cpu")

    # Load model
    net, (class_to_idx, signature, rgb_stats, *_) = fs_ops.load_model(
        device,
        args.network,
        tag=args.tag,
        epoch=args.epoch,
        inference=True,
        reparameterized=args.reparameterized,
    )
    if args.dynamic_size is True:
        net.set_dynamic_size()

    size = lib.get_size_from_signature(signature)
    input_channels = lib.get_channels_from_signature(signature)

    if args.dynamic_quantization is False:
        # Set calibration data for static activation observers
        full_dataset = make_image_dataset(
            [args.data_path],
            {},
            transforms=inference_preset(size, rgb_stats, 1.0),
            loader=get_image_loader(args.img_loader, input_channels),
        )
        num_calibration_samples = min(
            len(full_dataset),
            args.batch_size * args.num_calibration_batches,
        )
        if args.dynamic_batch is False:
            num_calibration_samples -= num_calibration_samples % args.batch_size
            if num_calibration_samples == 0:
                raise RuntimeError(f"Calibration dataset must contain at least {args.batch_size} samples")

        indices = torch.randperm(len(full_dataset))[:num_calibration_samples].tolist()
        calibration_dataset = Subset(full_dataset, indices=indices)
        calibration_data_loader = DataLoader(
            calibration_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=args.dynamic_batch is False,
        )

    # Quantization
    tic = time.time()
    quantizer = _build_quantizer(args.qbackend, args.per_channel, args.dynamic_quantization)
    if args.dynamic_quantization is False:
        calibration_iter = iter(calibration_data_loader)
        first_batch = next(calibration_iter)

    trace_batch_size = 2 if args.dynamic_batch is True else args.batch_size
    trace_size = args.trace_size if args.trace_size is not None else size
    sample_input = torch.randn(trace_batch_size, input_channels, *trace_size, device=device)

    signature["inputs"][0]["data_shape"][0] = sample_input.shape[0]
    example_inputs = (sample_input,)

    dynamic_shapes = None
    if args.dynamic_batch is True:
        if args.pte is True:
            batch_dim = torch.export.Dim("batch", min=1, max=args.max_batch_size)
        else:
            batch_dim = torch.export.Dim.DYNAMIC

        dynamic_shapes = {"x": {0: batch_dim}}

    if args.dynamic_size is True:
        if args.pte is True:
            height_dim = torch.export.Dim.DYNAMIC(max=args.max_size[0])
            width_dim = torch.export.Dim.DYNAMIC(max=args.max_size[1])
        else:
            height_dim = torch.export.Dim.DYNAMIC
            width_dim = torch.export.Dim.DYNAMIC

        if dynamic_shapes is None:
            dynamic_shapes = {"x": {}}

        dynamic_shapes["x"][2] = height_dim
        dynamic_shapes["x"][3] = width_dim

    with torch.no_grad():
        exported_net = torch.export.export(net, example_inputs, dynamic_shapes=dynamic_shapes, strict=True).module()
        prepared_net = prepare_pt2e(exported_net, quantizer)
        if args.dynamic_quantization is True:
            # Dynamic activations need no calibration, but weight observers need one pass.
            prepared_net(sample_input)

    if args.dynamic_quantization is False:
        with tqdm(total=num_calibration_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
            with torch.no_grad():
                for _, inputs, _ in itertools.chain([first_batch], calibration_iter):
                    inputs = inputs.to(device)
                    prepared_net(inputs)

                    # Update progress bar
                    progress.update(n=inputs.shape[0])

    with torch.no_grad():
        quantized_net = convert_pt2e(prepared_net)

        # Rebuild the module from the exported graph to drop the original float weights
        quantized_net = torch.export.export(
            quantized_net, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
        ).module()
        exported_quantized_net = torch.export.export(
            quantized_net, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
        )

    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to quantize model")

    if args.pte is True:
        logger.info(f"Lowering quantized model to PTE {model_path}...")
        _save_pte(exported_quantized_net, model_path, net.task, class_to_idx, signature, rgb_stats)
    else:
        logger.info(f"Saving quantized PT2 model {model_path}...")
        fs_ops.save_pt2(exported_quantized_net, model_path, net.task, class_to_idx, signature, rgb_stats)
