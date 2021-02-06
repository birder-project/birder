import argparse
from typing import Tuple

import mxnet as mx
import numpy as np

from birder.common import util


def export_onnx(model_path: str, epoch: int, input_shape: Tuple[int, int, int, int]) -> None:
    symbol_path = f"{model_path}-symbol.json"
    params_path = f"{model_path}-{epoch:04d}.params"

    mx.contrib.onnx.export_model(
        symbol_path, params_path, [input_shape], input_type=np.float32, onnx_file_path=f"{model_path}.onnx"
    )


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "export-onnx", help="export mxnet model to onnx format", formatter_class=util.ArgumentHelpFormatter
    )
    subparser.add_argument(
        "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "--size", type=int, default=None, help="trained image size (defaults to model signature)"
    )
    subparser.add_argument("-e", "--epoch", type=int, default=0, help="model checkpoint to load")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    classes = util.read_synset()

    network_name = f"{args.network}_{len(classes)}"
    model_path = util.get_model_path(network_name)
    if args.size is None:
        signature = util.read_signature(network_name)
        args.size = util.get_signature_size(signature)

    input_shape = (1, 3, args.size, args.size)

    export_onnx(model_path, args.epoch, input_shape)
