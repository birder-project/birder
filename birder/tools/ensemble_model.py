import argparse
import json
import logging
from typing import Any

import torch

from birder.common import cli


class Ensemble(torch.nn.Module):
    def __init__(self, module_list: torch.nn.ModuleList) -> None:
        super().__init__()
        self.module_list = module_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for module in self.module_list:
            outs.append(module(x))

        x = torch.stack(outs)
        x = torch.mean(x, dim=0)

        return x


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "ensemble-model",
        help="create an ensemble model from multiple torchscript models",
        description="create an ensemble model from multiple torchscript models",
        epilog=(
            "Usage examples:\n"
            "python3 tool.py ensemble-model --network convnext_v2_4_0 focalnet_3_0 swin_transformer_v2_1_0\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--networks", type=str, required=True, nargs="+", help="networks to ensemble")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    nets = torch.nn.ModuleList()
    class_to_idx_list = []
    signature_list = []
    rgb_values_list = []
    for network in args.networks:
        (net, class_to_idx, signature, rgb_values) = cli.load_model(device, network, inference=True, script=True)
        nets.append(net)
        class_to_idx_list.append(class_to_idx)
        signature_list.append(signature)
        rgb_values_list.append(rgb_values)

    # Ensure all have the same class to index definitions
    if [class_to_idx_list[0]] * len(class_to_idx_list) != class_to_idx_list:
        raise ValueError("All networks must have the same class to index definition")

    if [signature_list[0]] * len(signature_list) != signature_list:
        logging.warning(f"Networks signatures differ, using signature={signature_list[0]}")

    if [rgb_values_list[0]] * len(rgb_values_list) != rgb_values_list:
        logging.warning(f"Networks rgb values differ, using rgb values of {rgb_values_list[0]}")

    ensemble = Ensemble(nets)
    scripted_ensemble = torch.jit.script(ensemble)

    # Save model
    network_name = "ensemble"
    model_path = cli.model_path(network_name, script=True)
    logging.info(f"Saving TorchScript model checkpoint {model_path}...")
    torch.jit.save(
        scripted_ensemble,
        model_path,
        _extra_files={
            "class_to_idx": json.dumps(class_to_idx_list[0]),
            "signature": json.dumps(signature_list[0]),
            "rgb_values": json.dumps(rgb_values_list[0]),
        },
    )
