import argparse
import logging
from typing import Any
from typing import Optional

import torch

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common.lib import get_network_name
from birder.model_registry import registry
from birder.net.base import SignatureType

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals
def avg_models(
    network: str, tag: Optional[str], reparameterized: bool, epochs: list[int], accum_dtype: torch.dtype, force: bool
) -> None:
    device = torch.device("cpu")
    network_name = get_network_name(network, tag)
    state_list = []
    aux_data = {}
    for idx, epoch in enumerate(epochs):
        path = fs_ops.model_path(network_name, epoch=epoch)
        logger.info(f"Loading model from {path}...")

        model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        state_list.append(model_dict["state"])

        if idx == 0:
            logger.info(f"Copying signature from epoch {epoch}")
            for key in model_dict:
                if key in ("state", "signature"):
                    continue

                aux_data[key] = model_dict[key]
                logger.info(f"Copying {key} from epoch {epoch}")

            signature: SignatureType = model_dict["signature"]
            input_channels = lib.get_channels_from_signature(signature)
            num_classes = lib.get_num_labels_from_signature(signature)
            size = lib.get_size_from_signature(signature)

            net = registry.net_factory(network, num_classes, input_channels, size=size)
            if reparameterized is True:
                net.reparameterize_model()

            net.to(device)

    # Average state
    logger.info("Calculating averages...")
    avg_state = {}
    for state_name in state_list[0]:
        t0 = state_list[0][state_name]
        if torch.is_floating_point(t0) is True:
            params = torch.empty((len(state_list),) + t0.size(), dtype=accum_dtype)

            for idx, state in enumerate(state_list):
                params[idx] = state[state_name].to(accum_dtype)

            avg_state[state_name] = params.mean(dim=0).to(dtype=t0.dtype)
        else:
            # For int/bool buffers (e.g. num_batches_tracked / relative_position_index), averaging is not meaningful
            logger.info(f"Not averaging non-floating state entry: {state_name} (dtype={t0.dtype})")
            avg_state[state_name] = t0

    net.load_state_dict(avg_state)

    # Save model
    model_path = fs_ops.model_path(network_name, epoch=0)
    if model_path.exists() is True and force is False:
        logger.warning("Averaged model already exists... aborting")
        raise SystemExit(1)

    logger.info(f"Saving model checkpoint {model_path}...")
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
            "python -m birder.tools avg-model --network efficientnet_v2_m --epochs 290 295 300\n"
            "python -m birder.tools avg-model --network shufflenet_v2_2_0 --epochs 95 100 100 --accum-dtype float64\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2)"
    )
    subparser.add_argument("--epochs", type=int, nargs="+", metavar="N", help="epochs to average")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from the training phase)")
    subparser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    subparser.add_argument(
        "--accum-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="dtype used for averaging floating tensors",
    )
    subparser.add_argument("--force", action="store_true", help="override existing model")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    accum_dtype: torch.dtype = getattr(torch, args.accum_dtype)
    avg_models(args.network, args.tag, args.reparameterized, args.epochs, accum_dtype, args.force)
