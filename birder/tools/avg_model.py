import argparse
import logging
from typing import Any
from typing import Optional

import torch

from birder.common import cli
from birder.common.lib import get_network_name
from birder.core.net.base import SignatureType
from birder.model_registry import registry


def avg_models(network: str, net_param: Optional[float], tag: Optional[str], epochs: list[int], force: bool) -> None:
    device = torch.device("cpu")
    state_list = []
    aux_data = {}
    for idx, epoch in enumerate(epochs):
        network_name = get_network_name(network, net_param, tag)
        path = cli.model_path(network_name, epoch=epoch, pts=False)
        logging.info(f"Loading model from {path}...")

        model_dict: dict[str, Any] = torch.load(path, map_location=device)
        state_list.append(model_dict["state"])

        if idx == 0:
            logging.info(f"Copying signature from epoch {epoch}")
            for key in model_dict:
                if key in ("state", "signature"):
                    continue

                aux_data[key] = model_dict[key]
                logging.info(f"Copying {key} from epoch {epoch}")

            signature: SignatureType = model_dict["signature"]
            input_channels = signature["inputs"][0]["data_shape"][1]
            num_classes = signature["outputs"][0]["data_shape"][1]
            size = signature["inputs"][0]["data_shape"][2]

            net = registry.net_factory(network, input_channels, num_classes, net_param=net_param, size=size)
            net.to(device)

    # Average state
    logging.info("Calculating averages...")
    avg_state = {}
    for state_name in state_list[0]:
        params = torch.empty((len(state_list),) + state_list[0][state_name].size())

        for idx, state in enumerate(state_list):
            params[idx] = state[state_name]

        avg_state[state_name] = params.mean(axis=0)

    net.load_state_dict(avg_state)

    # Save model
    model_path = cli.model_path(network_name, epoch=0)
    if model_path.exists() is True and force is False:
        logging.warning("Averaged model already exists... aborting")
        raise SystemExit(1)

    logging.info(f"Saving model checkpoint {model_path}...")
    torch.save(
        {
            "state": net.state_dict(),
            "signature": signature,
            **aux_data,
        },
        model_path,
    )


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "avg-model",
        allow_abbrev=False,
        help="create weight average model from multiple trained models",
        description="create weight average model from multiple trained models",
        epilog=(
            "Usage examples:\n"
            "python tool.py avg-model --network efficientnet_v2_m --epochs 290 295 300\n"
            "python tool.py avg-model --network shufflenet_v2 --net-param 2 --epochs 95 100 100\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2)"
    )
    subparser.add_argument(
        "-p", "--net-param", type=float, help="network specific parameter, required for most networks"
    )
    subparser.add_argument("--epochs", type=int, nargs="+", help="epochs to average")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    subparser.add_argument("--force", action="store_true", help="override existing model")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    avg_models(args.network, args.net_param, args.tag, args.epochs, args.force)
