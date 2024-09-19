import argparse
import json
import logging
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
            "python -m birder.tools convert-model -n efficientnet_v2 -p 1 -e 0 --lite\n"
            "python -m birder.tools convert-model --network faster_rcnn --backbone resnext "
            "--backbone-param 101 -e 0 --pts\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to load (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required for most networks"
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
        help="network specific parameter, required by most networks (for the backbone)",
    )
    subparser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    subparser.add_argument("-e", "--epoch", type=int, help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")

    format_group = subparser.add_mutually_exclusive_group(required=True)
    format_group.add_argument("-r", "--reparameterize", default=False, action="store_true", help="reparameterize model")
    format_group.add_argument("--pts", default=False, action="store_true", help="convert to TorchScript model")
    format_group.add_argument(
        "--lite", default=False, action="store_true", help="convert to lite interpreter version model"
    )
    format_group.add_argument(
        "--pt2", default=False, action="store_true", help="convert to standardized model representation"
    )
    format_group.add_argument("--onnx", default=False, action="store_true", help="convert to ONNX format")

    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    # Load model
    device = torch.device("cpu")
    signature: SignatureType | DetectionSignatureType
    if args.backbone is None:
        (net, class_to_idx, signature, rgb_values) = fs_ops.load_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            epoch=args.epoch,
            inference=True,
            pts=False,
        )
        network_name = lib.get_network_name(args.network, net_param=args.net_param, tag=args.tag)

    else:
        (net, class_to_idx, signature, rgb_values) = fs_ops.load_detection_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_tag=args.backbone_tag,
            epoch=args.epoch,
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

    model_path = fs_ops.model_path(
        network_name, epoch=args.epoch, pts=args.pts, lite=args.lite, pt2=args.pt2, onnx=args.onnx
    )
    logging.info(f"Saving converted model {model_path}...")
    if args.reparameterize is True:
        if reparameterize_available(net) is False:
            logging.error("Reparameterize not supported for this network")
        else:
            net.reparameterize_model()
            network_name = lib.get_network_name(network_name, net_param=None, tag="reparameterized")
            fs_ops.checkpoint_model(
                network_name,
                args.epoch,
                net,
                signature=signature,
                class_to_idx=class_to_idx,
                rgb_values=rgb_values,
                optimizer=None,
                scheduler=None,
                scaler=None,
            )

    elif args.lite is True:
        scripted_module = torch.jit.script(net)
        optimized_scripted_module = optimize_for_mobile(scripted_module)
        optimized_scripted_module._save_for_lite_interpreter(  # pylint: disable=protected-access
            str(model_path),
            _extra_files={
                "task": net.task,
                "class_to_idx": json.dumps(class_to_idx),
                "signature": json.dumps(signature),
                "rgb_values": json.dumps(rgb_values),
            },
        )

    elif args.pt2 is True:
        signature["inputs"][0]["data_shape"][0] = 2  # Set batch size
        sample_shape = signature["inputs"][0]["data_shape"]
        batch_dim = torch.export.Dim("batch", min=1)
        exported_net = torch.export.export(
            net, (torch.randn(*sample_shape, device=device),), dynamic_shapes={"x": {0: batch_dim}}
        )
        fs_ops.save_pt2(exported_net, model_path, net.task, class_to_idx, signature, rgb_values)

    elif args.onnx is True:
        signature["inputs"][0]["data_shape"][0] = 1  # Set batch size
        sample_shape = signature["inputs"][0]["data_shape"]
        torch.onnx.export(
            net,
            torch.randn(sample_shape),
            model_path,
            export_params=True,
            opset_version=16,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        signature["inputs"][0]["data_shape"][0] = 0

        with open(f"{model_path}_class_to_idx.json", "w", encoding="utf-8") as handle:
            json.dump(class_to_idx, handle, indent=2)

        with open(f"{model_path}_signature.json", "w", encoding="utf-8") as handle:
            json.dump(signature, handle, indent=2)

        with open(f"{model_path}_rgb_values.json", "w", encoding="utf-8") as handle:
            json.dump(rgb_values, handle, indent=2)

        # Test exported model
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model, full_check=True)

    elif args.pts is True:
        scripted_module = torch.jit.script(net)
        fs_ops.save_pts(scripted_module, model_path, net.task, class_to_idx, signature, rgb_values)
