import argparse
import logging
from typing import List

import mxnet as mx

from birder.common import util
from birder.common.net import create_model


def avg_model(model_path: str, epochs: List[int], size: int) -> mx.module.Module:
    arg_params_list = []
    aux_params_list = []

    logging.info("Loading models")
    for epoch in epochs:
        (sym, arg_params, aux_params) = mx.model.load_checkpoint(model_path, epoch)
        arg_params_list.append(arg_params)
        aux_params_list.append(aux_params)

    for i in range(len(arg_params_list) - 1):
        assert arg_params_list[i].keys() == arg_params_list[i + 1].keys()
        assert aux_params_list[i].keys() == aux_params_list[i + 1].keys()

    logging.info("Calculating averages")

    # Average arg_params
    avg_arg_params = {}
    for layer_name in arg_params_list[0].keys():
        layer_params = mx.nd.empty((len(arg_params_list),) + arg_params_list[0][layer_name].shape)
        for i, params in enumerate(arg_params_list):
            layer_params[i] = params[layer_name]

        avg_arg_params[layer_name] = layer_params.mean(axis=0)

    # Average aux_params
    avg_aux_params = {}
    for layer_name in aux_params_list[0].keys():
        layer_params = mx.nd.empty((len(aux_params_list),) + aux_params_list[0][layer_name].shape)
        for i, params in enumerate(aux_params_list):
            layer_params[i] = params[layer_name]

        avg_aux_params[layer_name] = layer_params.mean(axis=0)

    return create_model(sym, avg_arg_params, avg_aux_params, mx.cpu(), (size, size), for_training=True)


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "avg-model",
        help="create weight average model from multiple trained models",
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2_50)"
    )
    subparser.add_argument(
        "--size", type=int, default=None, help="trained image size (defaults to model signature)"
    )
    subparser.add_argument("--epochs", type=int, nargs="+", help="epochs to average")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    classes = util.read_synset()

    network_name = f"{args.network}_{len(classes)}"
    model_path = util.get_model_path(network_name)
    if args.size is None:
        signature = util.read_signature(network_name)
        args.size = util.get_signature_size(signature)

    model = avg_model(model_path, args.epochs, args.size)
    model.save_checkpoint(model_path, 0, save_optimizer_states=False)
