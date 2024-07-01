import argparse
import json
import logging
from typing import Any
from typing import Union

import onnx
import onnx.checker
import torch
import torch.onnx
from torch.utils.mobile_optimizer import optimize_for_mobile

from birder.common import cli
from birder.common import lib
from birder.core.net.base import DetectorBackbone
from birder.core.net.base import SignatureType
from birder.core.net.base import network_names_filter
from birder.core.net.detection.base import DetectionSignatureType


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "script-model",
        help="convert pytorch model to torchscript model",
        description="convert pytorch model to torchscript model",
        epilog=(
            "Usage examples:\n"
            "python3 tool.py script-model --network shufflenet_v2 --net-param 2 --epoch 200\n"
            "python3 tool.py script-model --network squeezenet --epoch 100\n"
            "python3 tool.py script-model --network densenet -p 121 -e 100 --pt2\n"
            "python3 tool.py script-model -n efficientnet_v2 -p 1 -e 200 --lite\n"
            "python3 tool.py script-model --network faster_rcnn --backbone resnext "
            "--backbone-param 101 -e 0\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to load (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required by most networks"
    )
    subparser.add_argument(
        "--backbone",
        type=str,
        choices=network_names_filter(DetectorBackbone),
        help="the neural network to used as backbone",
    )
    subparser.add_argument(
        "--backbone-param",
        type=float,
        help="network specific parameter, required by most networks (for the backbone)",
    )
    subparser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    subparser.add_argument("-e", "--epoch", type=int, default=0, help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument("--lite", default=False, action="store_true", help="lite interpreter version model")
    subparser.add_argument("--pt2", default=False, action="store_true", help="standardized model representation")
    subparser.add_argument("--onnx", default=False, action="store_true", help="export to onnx format")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    # Load model
    device = torch.device("cpu")
    signature: Union[SignatureType, DetectionSignatureType]
    if args.backbone is None:
        (net, class_to_idx, signature, rgb_values) = cli.load_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            epoch=args.epoch,
            inference=True,
            script=False,
        )
        network_name = lib.get_network_name(args.network, net_param=args.net_param, tag=args.tag)

    else:
        (net, class_to_idx, signature, rgb_values) = cli.load_detection_model(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_tag=args.backbone_tag,
            epoch=args.epoch,
            inference=True,
            script=False,
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

    model_path = cli.model_path(
        network_name, epoch=args.epoch, script=True, lite=args.lite, pt2=args.pt2, onnx=args.onnx
    )
    logging.info(f"Saving converted model {model_path}...")
    if args.lite is True:
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
        torch.export.save(
            exported_net,
            model_path,
            extra_files={
                "task": net.task,
                "class_to_idx": json.dumps(class_to_idx),
                "signature": json.dumps(signature),
                "rgb_values": json.dumps(rgb_values),
            },
        )

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

    else:
        scripted_module = torch.jit.script(net)
        torch.jit.save(
            scripted_module,
            model_path,
            _extra_files={
                "task": net.task,
                "class_to_idx": json.dumps(class_to_idx),
                "signature": json.dumps(signature),
                "rgb_values": json.dumps(rgb_values),
            },
        )
