import argparse
import json
import logging
from pathlib import Path
from typing import Any

import onnx
import onnx.checker
import torch
import torch.onnx
from torch.utils.mobile_optimizer import optimize_for_mobile

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import SignatureType
from birder.net.base import reparameterize_available
from birder.net.detection.base import DetectionSignatureType
from birder.transforms.classification import RGBType

try:
    import safetensors.torch

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def reparameterize(
    net: torch.nn.Module,
    signature: SignatureType | DetectionSignatureType,
    class_to_idx: dict[str, int],
    rgb_stats: RGBType,
    epoch: int,
    network_name: str,
) -> None:
    if reparameterize_available(net) is False:
        logging.error("Reparameterize not supported for this network")
    else:
        net.reparameterize_model()
        network_name = lib.get_network_name(network_name, net_param=None, tag="reparameterized")
        fs_ops.checkpoint_model(
            network_name,
            epoch,
            net,
            signature=signature,
            class_to_idx=class_to_idx,
            rgb_stats=rgb_stats,
            optimizer=None,
            scheduler=None,
            scaler=None,
        )


def pt2_export(
    net: torch.nn.Module,
    signature: SignatureType | DetectionSignatureType,
    class_to_idx: dict[str, int],
    rgb_stats: RGBType,
    device: torch.device,
    model_path: str | Path,
) -> None:
    signature["inputs"][0]["data_shape"][0] = 2  # Set batch size
    sample_shape = signature["inputs"][0]["data_shape"]
    batch_dim = torch.export.Dim("batch", min=1)
    exported_net = torch.export.export(
        net, (torch.randn(*sample_shape, device=device),), dynamic_shapes={"x": {0: batch_dim}}
    )
    fs_ops.save_pt2(exported_net, model_path, net.task, class_to_idx, signature, rgb_stats)


def onnx_export(
    net: torch.nn.Module,
    signature: SignatureType | DetectionSignatureType,
    class_to_idx: dict[str, int],
    rgb_stats: RGBType,
    model_path: str | Path,
    dynamo: bool,
    trace: bool,
) -> None:
    signature["inputs"][0]["data_shape"][0] = 1  # Set batch size
    sample_shape = signature["inputs"][0]["data_shape"]

    if dynamo is False:
        if trace is False:
            scripted_module = torch.jit.script(net)
        else:
            scripted_module = net

        torch.onnx.export(
            scripted_module,
            torch.randn(sample_shape),
            str(model_path),
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    else:
        onnx_program = torch.onnx.dynamo_export(net, torch.randn(sample_shape))
        onnx_program.save(str(model_path))

    signature["inputs"][0]["data_shape"][0] = 0

    logging.info("Saving class to index json...")
    with open(f"{model_path}_class_to_idx.json", "w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2)

    model_config = lib.get_network_config(net, signature, rgb_stats)
    logging.info("Saving model config json...")
    with open(f"{model_path}_config.json", "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)

    # Test exported model
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model, full_check=True)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "convert-model",
        allow_abbrev=False,
        help="convert PyTorch model to various formats",
        description="convert PyTorch model to various formats",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools convert-model --network shufflenet_v2 --net-param 2 --epoch 200 --pts\n"
            "python -m birder.tools convert-model --network squeezenet --epoch 100 --onnx\n"
            "python -m birder.tools convert-model -n mobilevit_v2 -p 1.5 -t intermediate -e 80 --pt2\n"
            "python -m birder.tools convert-model -n efficientnet_v2_m -e 0 --lite\n"
            "python -m birder.tools convert-model --network faster_rcnn --backbone resnext "
            "--backbone-param 101 -e 0 --pts\n"
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
    subparser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="trace instead of script (applies only to TorchScript conversions)",
    )
    subparser.add_argument("--force", action="store_true", help="override existing model")

    format_group = subparser.add_mutually_exclusive_group(required=True)
    format_group.add_argument("--resize", type=int, help="resize model (pt)")
    format_group.add_argument("--reparameterize", default=False, action="store_true", help="reparameterize model")
    format_group.add_argument("--pts", default=False, action="store_true", help="convert to TorchScript model")
    format_group.add_argument(
        "--lite", default=False, action="store_true", help="convert to lite TorchScript interpreter version model"
    )
    format_group.add_argument(
        "--pt2", default=False, action="store_true", help="convert to standardized model representation"
    )
    format_group.add_argument("--st", default=False, action="store_true", help="convert to Safetensors")
    format_group.add_argument("--onnx", default=False, action="store_true", help="convert to ONNX format")
    format_group.add_argument(
        "--onnx-dynamo", default=False, action="store_true", help="convert to ONNX format using TorchDynamo"
    )

    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    assert args.trace is False or (args.trace is True and (args.pts is True or args.lite is True or args.onnx is True))

    # Load model
    device = torch.device("cpu")
    signature: SignatureType | DetectionSignatureType
    if args.backbone is None:
        (net, class_to_idx, signature, rgb_stats) = fs_ops.load_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            epoch=args.epoch,
            new_size=args.resize,
            inference=True,
            reparameterized=args.reparameterized,
        )
        network_name = lib.get_network_name(args.network, net_param=args.net_param, tag=args.tag)

    else:
        (net, class_to_idx, signature, rgb_stats) = fs_ops.load_detection_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_tag=args.backbone_tag,
            epoch=args.epoch,
            new_size=args.resize,
            inference=True,
            pts=False,
        )
        network_name = lib.get_detection_network_name(
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_tag=args.backbone_tag,
        )

    net.eval()

    if args.resize is not None:
        network_name = f"{network_name}_{args.resize}px"

    model_path = fs_ops.model_path(
        network_name,
        epoch=args.epoch,
        pts=args.pts,
        lite=args.lite,
        pt2=args.pt2,
        st=args.st,
        onnx=args.onnx or args.onnx_dynamo,
    )
    if model_path.exists() is True and args.force is False and args.reparameterize is False:
        logging.warning("Converted model already exists... aborting")
        raise SystemExit(1)

    logging.info(f"Saving converted model {model_path}...")
    if args.resize is not None:
        net.adjust_size(args.resize)
        signature["inputs"][0]["data_shape"][2] = args.resize
        signature["inputs"][0]["data_shape"][3] = args.resize
        fs_ops.checkpoint_model(
            network_name,
            args.epoch,
            net,
            signature,
            class_to_idx,
            rgb_stats,
            optimizer=None,
            scheduler=None,
            scaler=None,
        )

    elif args.reparameterize is True:
        reparameterize(net, signature, class_to_idx, rgb_stats, args.epoch, network_name)

    elif args.lite is True:
        if args.trace is True:
            sample_shape = [1] + signature["inputs"][0]["data_shape"][1:]  # C, H, W
            scripted_module = torch.jit.trace(net, example_inputs=torch.randn(sample_shape))
            optimized_scripted_module = scripted_module
        else:
            scripted_module = torch.jit.script(net)
            optimized_scripted_module = optimize_for_mobile(scripted_module)

        optimized_scripted_module._save_for_lite_interpreter(  # pylint: disable=protected-access
            str(model_path),
            _extra_files={
                "task": net.task,
                "class_to_idx": json.dumps(class_to_idx),
                "signature": json.dumps(signature),
                "rgb_stats": json.dumps(rgb_stats),
            },
        )

    elif args.pt2 is True:
        pt2_export(net, signature, class_to_idx, rgb_stats, device, model_path)

    elif args.st is True:
        assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
        safetensors.torch.save_model(
            net,
            str(model_path),
            {
                "task": net.task,
                "class_to_idx": json.dumps(class_to_idx),
                "signature": json.dumps(signature),
                "rgb_stats": json.dumps(rgb_stats),
            },
        )

    elif args.onnx is True or args.onnx_dynamo:
        onnx_export(net, signature, class_to_idx, rgb_stats, model_path, args.onnx_dynamo, args.trace)

    elif args.pts is True:
        if args.trace is True:
            sample_shape = [1] + signature["inputs"][0]["data_shape"][1:]  # C, H, W
            scripted_module = torch.jit.trace(net, example_inputs=torch.randn(sample_shape))
        else:
            scripted_module = torch.jit.script(net)

        fs_ops.save_pts(scripted_module, model_path, net.task, class_to_idx, signature, rgb_stats)
