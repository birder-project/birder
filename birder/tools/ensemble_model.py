import argparse
import json
import logging
from typing import List

import mxnet as mx

from birder.common import util
from birder.common.net import create_model
from birder.common.preprocess import DEFAULT_RGB
from birder.common.preprocess import IMAGENET_RGB


def ensemble_model(models_path: List[str], size: int) -> mx.module.Module:
    epoch = 0
    symbol_list = []
    arg_params_dict = {}
    aux_params_dict = {}

    logging.info("Loading models")
    for model_idx, model_path in enumerate(models_path):
        (sym, arg_params, aux_params) = mx.model.load_checkpoint(model_path, epoch)
        sym_dict = json.loads(sym.tojson())

        # Change model layer names to avoid duplicates
        for node in sym_dict["nodes"]:
            if node["name"] in ["data", "softmax_label"]:
                continue

            node["name"] = f"{model_idx}_{node['name']}"

        sym = mx.sym.load_json(json.dumps(sym_dict))

        # Change arg params layer names
        new_arg_params = {}
        for layer_name, params in arg_params.items():
            new_arg_params[f"{model_idx}_{layer_name}"] = params

        # Change aux params layer names
        new_aux_params = {}
        for layer_name, params in aux_params.items():
            new_aux_params[f"{model_idx}_{layer_name}"] = params

        symbol_list.append(sym)
        arg_params_dict.update(new_arg_params)
        aux_params_dict.update(new_aux_params)

    ensemble_model_symbol = mx.sym.Concat(*symbol_list, dim=0)
    ensemble_model_symbol = mx.sym.mean(ensemble_model_symbol, axis=0)
    ensemble_model_symbol_dict = json.loads(ensemble_model_symbol.tojson())  # pylint: disable=no-member

    # Remove duplicate node inputs and outputs
    inputs = []
    labels = []
    for idx, node in enumerate(ensemble_model_symbol_dict["nodes"]):
        if node["name"] == "data":
            inputs.append(idx)

        if node["name"] == "softmax_label":
            labels.append(idx)

    for idx, node in enumerate(ensemble_model_symbol_dict["nodes"]):
        for cell in node["inputs"]:
            if cell[0] in inputs:
                cell[0] = inputs[0]

            if cell[0] in labels:
                cell[0] = labels[0]

    # Final symbol
    ensemble_model_symbol = mx.sym.load_json(json.dumps(ensemble_model_symbol_dict))

    # Print model summary
    mx.visualization.print_summary(ensemble_model_symbol, shape={"data": (1, 3, size, size)})

    return create_model(
        ensemble_model_symbol, arg_params_dict, aux_params_dict, mx.cpu(), (size, size), for_training=False
    )


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "ensemble-model",
        help="create ensemble model from multiple trained models",
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--network", type=str, required=True, nargs="+", help="the neural network to use (i.e. resnet_v2_50)"
    )
    subparser.add_argument("--size", type=int, default=224, help="ensemble model input image size")
    subparser.add_argument(
        "--rgb", default=False, action="store_true", help="use pre-calculated mean and std rgb values"
    )
    subparser.add_argument(
        "--rgb-imagenet", default=False, action="store_true", help="use imagenet mean and std rgb values"
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    classes = util.read_synset()

    models_path = []
    for network in args.network:
        network_name = f"{network}_{len(classes)}"
        models_path.append(util.get_model_path(network_name))

    signature = util.read_signature(network_name)
    if args.size is None:
        args.size = util.get_signature_size(signature)

    model = ensemble_model(models_path, args.size)
    network_name = f"ensemble_{len(classes)}"

    if args.rgb is True:
        rgb_values = util.read_rgb()

    elif args.rgb_imagenet is True:
        rgb_values = IMAGENET_RGB

    else:
        rgb_values = DEFAULT_RGB

    util.write_signature(network_name, args.size, rgb_values)
    model_path = util.get_model_path(network_name)
    model.save_checkpoint(model_path, 0, save_optimizer_states=False)
