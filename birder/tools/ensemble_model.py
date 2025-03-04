import argparse
import logging
from typing import Any

import torch

from birder.common import cli
from birder.common import fs_ops

logger = logging.getLogger(__name__)


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
        allow_abbrev=False,
        help="create an ensemble model from multiple models",
        description="create an ensemble model from multiple models",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools ensemble-model --networks convnext_v2_4_0 focalnet_3_0 "
            "swin_transformer_v2_1_0 --pts\n"
            "python -m birder.tools ensemble-model --networks mobilevit_v2_1.5_intermediate_80 "
            "edgevit_2_intermediate_100 --pt2"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--networks", type=str, required=True, nargs="+", help="networks to ensemble")
    subparser.add_argument("--force", action="store_true", help="override existing model")

    format_group = subparser.add_mutually_exclusive_group(required=True)
    format_group.add_argument("--pts", default=False, action="store_true", help="ensemble TorchScript models")
    format_group.add_argument("--pt2", default=False, action="store_true", help="ensemble pt2 models")

    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    nets = torch.nn.ModuleList()
    class_to_idx_list = []
    signature_list = []
    rgb_stats_list = []
    for network in args.networks:
        (net, model_info) = fs_ops.load_model(device, network, inference=True, pts=args.pts, pt2=args.pt2)
        nets.append(net)
        class_to_idx_list.append(model_info.class_to_idx)
        signature_list.append(model_info.signature)
        rgb_stats_list.append(model_info.rgb_stats)

    # Ensure all have the same class to index definitions
    if [class_to_idx_list[0]] * len(class_to_idx_list) != class_to_idx_list:
        raise ValueError("All networks must have the same class to index definition")

    if [signature_list[0]] * len(signature_list) != signature_list:
        logger.warning(f"Networks signatures differ, using signature={signature_list[0]}")

    if [rgb_stats_list[0]] * len(rgb_stats_list) != rgb_stats_list:
        logger.warning(f"Networks rgb values differ, using rgb values of {rgb_stats_list[0]}")

    signature = signature_list[0]
    class_to_idx = class_to_idx_list[0]
    rgb_stats = rgb_stats_list[0]
    ensemble = Ensemble(nets)

    network_name = "ensemble"
    model_path = fs_ops.model_path(network_name, pts=args.pts, pt2=args.pt2)
    if model_path.exists() is True and args.force is False:
        logger.warning("Ensemble already exists... aborting")
        raise SystemExit(1)

    logger.info(f"Saving model {model_path}...")
    if args.pt2 is True:
        signature["inputs"][0]["data_shape"][0] = 2  # Set batch size
        sample_shape = signature["inputs"][0]["data_shape"]
        batch_dim = torch.export.Dim("batch", min=1)
        exported_net = torch.export.export(
            ensemble, (torch.randn(*sample_shape, device=device),), dynamic_shapes={"x": {0: batch_dim}}
        )

        # Save model
        fs_ops.save_pt2(exported_net, model_path, nets[0].task, class_to_idx, signature, rgb_stats)

    elif args.pts is True:
        scripted_ensemble = torch.jit.script(ensemble)

        # Save model
        fs_ops.save_pts(scripted_ensemble, model_path, nets[0].task, class_to_idx, signature, rgb_stats)
