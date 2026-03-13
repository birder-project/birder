import argparse
import logging
import time
from typing import Any
from typing import Optional

import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

import birder
from birder.adversarial.base import Attack
from birder.adversarial.deepfool import DeepFool
from birder.adversarial.fgsm import FGSM
from birder.adversarial.pgd import PGD
from birder.adversarial.simba import SimBA
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import inference_preset
from birder.inference.data_parallel import InferenceDataParallel
from birder.model_registry import Task

logger = logging.getLogger(__name__)


def _resolve_pretrained_models(args: argparse.Namespace) -> list[str]:
    if args.filter is None:
        return []

    model_list = birder.list_pretrained_models(args.filter, task=Task.IMAGE_CLASSIFICATION)
    if len(model_list) == 0:
        logger.warning(f"No pretrained models matched filter {args.filter!r}")

    return model_list


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("Adversarial", style="dim")
    table.add_column("Attack", justify="right")
    table.add_column("Eps", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Step Size", justify="right")
    table.add_column("Clean Acc", justify="right")
    table.add_column("Adv Acc", justify="right")
    table.add_column("Acc Drop", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Skipped", justify="right")

    for result in results:
        step_size = result["step_size"]
        table.add_row(
            str(result["network"]),
            str(result["attack_method"]),
            f"{float(result['epsilon']):g}",
            str(result["steps"]),
            f"{float(step_size):g}",
            f"{float(result['clean_accuracy']):.4f}",
            f"{float(result['accuracy']):.4f}",
            f"{float(result['accuracy_drop']):.4f}",
            str(result["num_samples"]),
            str(result["num_skipped_unlabeled"]),
        )

    console.print(table)


def _build_attack(
    method: str,
    net: torch.nn.Module,
    rgb_stats: RGBType,
    eps: float,
    steps: int,
    step_size: Optional[float],
    deepfool_num_classes: int,
) -> Attack:
    if method == "fgsm":
        return FGSM(net, eps=eps, rgb_stats=rgb_stats)

    if method == "pgd":
        return PGD(
            net,
            eps=eps,
            steps=steps,
            step_size=step_size,
            random_start=False,
            rgb_stats=rgb_stats,
        )

    if method == "deepfool":
        return DeepFool(net, num_classes=deepfool_num_classes, overshoot=0.02, max_iter=steps, rgb_stats=rgb_stats)

    if method == "simba":
        return SimBA(
            net,
            step_size=step_size if step_size is not None else eps,
            max_iter=steps,
            rgb_stats=rgb_stats,
        )

    raise ValueError(f"Unsupported attack method '{method}'")


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def evaluate_adversarial_robustness(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    elif args.mps is True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)

        logger.info(f"Using device {device}")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
        logger.debug(f"AMP: {args.amp}, AMP dtype: {amp_dtype}")
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    pretrained_models = _resolve_pretrained_models(args)
    model_runs: list[tuple[str, bool]] = [(model_name, False) for model_name in pretrained_models]
    if args.network is not None:
        model_runs.append((lib.get_network_name(args.network, tag=args.tag), True))

    all_results: list[dict[str, Any]] = []
    total_num_samples = 0
    total_tic = time.time()
    for network_name, is_checkpoint in model_runs:
        if is_checkpoint is False:
            net, model_info = birder.load_pretrained_model(network_name, inference=True, device=device)
        else:
            net, model_info = fs_ops.load_model(
                device,
                args.network,
                tag=args.tag,
                epoch=args.epoch,
                new_size=args.size,
                inference=True,
                reparameterized=args.reparameterized,
            )

        if args.parallel is True and torch.cuda.device_count() > 1:
            net = InferenceDataParallel(net)

        class_to_idx = model_info.class_to_idx
        rgb_stats = model_info.rgb_stats
        if args.size is None:
            size = lib.get_size_from_signature(model_info.signature)
        else:
            size = args.size

        transform = inference_preset(size, rgb_stats, args.center_crop, args.simple_crop)

        if args.wds is True:
            wds_path: str | list[str]
            if args.wds_info is not None:
                wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
                if args.wds_size is not None:
                    dataset_size = args.wds_size
            else:
                wds_path, dataset_size = prepare_wds_args(args.data_path[0], args.wds_size, device)

            num_samples = dataset_size
            dataset = make_wds_dataset(
                wds_path,
                dataset_size=dataset_size,
                shuffle=False,
                samples_names=True,
                transform=transform,
            )
            dataloader = make_wds_loader(
                dataset,
                args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=None,
                collate_fn=None,
                world_size=1,
                pin_memory=False,
                exact=True,
            )
        else:
            dataset = make_image_dataset(args.data_path, class_to_idx, transforms=transform)
            num_samples = len(dataset)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        total_num_samples += num_samples
        attack = _build_attack(
            args.method, net, rgb_stats, args.eps, args.steps, args.step_size, args.deepfool_num_classes
        )

        clean_correct = 0
        adv_correct = 0
        total = 0
        skipped_unlabeled = 0
        with tqdm(total=num_samples, unit="images", leave=False) as progress:
            for _, inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size = inputs.size(0)

                valid_mask = targets != settings.NO_LABEL
                num_valid = valid_mask.sum().item()
                skipped_unlabeled += batch_size - num_valid
                if num_valid == 0:
                    progress.update(batch_size)
                    continue

                inputs = inputs[valid_mask]
                targets = targets[valid_mask]

                with torch.no_grad():
                    with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
                        clean_logits = net(inputs)

                    clean_preds = clean_logits.argmax(dim=1)
                    clean_correct += (clean_preds == targets).sum().item()

                result = attack(inputs, target=None)
                adv_logits = result.adv_logits
                adv_preds = adv_logits.argmax(dim=1)
                adv_correct += (adv_preds == targets).sum().item()

                total += num_valid
                progress.update(batch_size)

        if total == 0:
            raise RuntimeError(f"No labeled samples found (all labels are {settings.NO_LABEL})")

        if skipped_unlabeled > 0:
            logger.warning(f"Skipped {skipped_unlabeled} unlabeled samples (label={settings.NO_LABEL})")

        clean_accuracy = clean_correct / total
        adv_accuracy = adv_correct / total
        accuracy_drop = clean_accuracy - adv_accuracy

        logger.info(
            f"{network_name}: clean={clean_accuracy:.4f}, adv={adv_accuracy:.4f}, drop={accuracy_drop:.4f} "
            f"(evaluated on {total} labeled samples)"
        )

        output = {
            "network": network_name,
            "tag": args.tag if is_checkpoint is True else None,
            "epoch": args.epoch if is_checkpoint is True else None,
            "method": "adversarial",
            "attack_method": args.method,
            "epsilon": args.eps,
            "steps": args.steps,
            "step_size": args.step_size,
            "deepfool_num_classes": args.deepfool_num_classes,
            "accuracy": adv_accuracy,
            "clean_accuracy": clean_accuracy,
            "accuracy_drop": accuracy_drop,
            "num_samples": total,
            "num_skipped_unlabeled": skipped_unlabeled,
        }
        all_results.append(output)

    _print_summary_table(all_results)
    if args.dry_run is False:
        output_dir = settings.RESULTS_DIR.joinpath(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath(f"adversarial_{args.method}_eps{args.eps}.csv")
        pl.DataFrame(all_results).write_csv(output_path)
        logger.info(f"Results saved to {output_path}")

    total_toc = time.time()
    total_elapsed = total_toc - total_tic
    logger.info(
        f"Adversarial evaluation completed in {lib.format_duration(total_elapsed)} "
        f"for {len(model_runs)} models and {total_num_samples:,} sampled inputs"
    )


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "adversarial",
        allow_abbrev=False,
        help="evaluate adversarial robustness of a model on a dataset",
        description="evaluate adversarial robustness of a model on a dataset",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval adversarial --filter '*il-all*' --method fgsm --eps 0.01 "
            "--gpu data/validation_il-all_packed\n"
            "python -m birder.eval adversarial --filter '*il-all*' -n rope_vit_reg4_b14 -t capi-raw336px "
            "--method fgsm --eps 0.01 --gpu data/validation_il-all_packed\n"
            "python -m birder.eval adversarial -n rope_vit_reg4_b14 -t capi-raw336px -e 0 --method fgsm --eps 0.01 "
            "--batch-size 128 --amp --amp-dtype bfloat16 --gpu --dry-run data/validation_eu-common_packed\n"
            "python -m birder.eval adversarial -n vovnet_v2_39 -t il-all --method pgd --batch-size 4 "
            "--gpu --gpu-id 1 --fast-matmul data/validation_il-all_packed\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--filter", type=str, help="pretrained models to evaluate (fnmatch type filter)")
    subparser.add_argument("-n", "--network", type=str, help="checkpoint network to evaluate")
    subparser.add_argument("-t", "--tag", type=str, help="checkpoint model tag")
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="checkpoint model epoch")
    subparser.add_argument(
        "--reparameterized", default=False, action="store_true", help="load reparameterized checkpoint model"
    )
    subparser.add_argument(
        "--method",
        type=str,
        choices=["fgsm", "pgd", "deepfool", "simba"],
        help="adversarial attack method",
    )
    subparser.add_argument("--eps", type=float, default=0.007, help="perturbation budget in pixel space [0, 1]")
    subparser.add_argument("--steps", type=int, default=10, help="number of iterations for iterative attacks")
    subparser.add_argument("--step-size", type=float, help="step size in pixel space (defaults to eps/steps for PGD)")
    subparser.add_argument(
        "--deepfool-num-classes", type=int, default=10, help="number of top classes to consider for DeepFool"
    )
    subparser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    subparser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    subparser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    subparser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    subparser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    subparser.add_argument(
        "-j", "--num-workers", type=int, default=8, metavar="N", help="number of preprocessing workers"
    )
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    subparser.add_argument(
        "--simple-crop",
        default=False,
        action="store_true",
        help="use a simple crop that preserves aspect ratio but may trim parts of the image",
    )
    subparser.add_argument(
        "--dir",
        type=str,
        default="adversarial_robustness",
        help="place all outputs in a sub-directory (relative to results)",
    )
    subparser.add_argument("--dry-run", default=False, action="store_true", help="skip saving results to file")
    subparser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    subparser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    subparser.add_argument(
        "--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device"
    )
    subparser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpus")
    subparser.add_argument("--wds", default=False, action="store_true", help="evaluate a webdataset directory")
    subparser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds dataset")
    subparser.add_argument(
        "--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths"
    )
    subparser.add_argument(
        "--wds-split", type=str, default="validation", metavar="NAME", help="wds dataset split to load"
    )
    subparser.add_argument("data_path", nargs="*", help="data files path (directories and files)")
    subparser.set_defaults(func=main)


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    if args.filter is None and args.network is None:
        raise cli.ValidationError("At least one model selector is required via --filter and/or --network")
    if args.method is None:
        raise cli.ValidationError("--method is required")
    if args.network is None and args.tag is not None:
        raise cli.ValidationError("--tag requires --network")
    if args.network is None and args.epoch is not None:
        raise cli.ValidationError("--epoch requires --network")
    if args.network is None and args.reparameterized is True:
        raise cli.ValidationError("--reparameterized requires --network")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.parallel is True and args.gpu is False:
        raise cli.ValidationError("--parallel requires --gpu to be set")

    if args.wds is False and len(args.data_path) == 0:
        raise cli.ValidationError("Must provide at least one data source: DATA_PATH positional argument or --wds")

    if args.wds is True:
        if args.wds_info is None and len(args.data_path) == 0:
            raise cli.ValidationError("--wds requires a data path unless --wds-info is provided")
        if len(args.data_path) > 1:
            raise cli.ValidationError(
                f"--wds can have at most 1 DATA_PATH positional argument, got {len(args.data_path)}"
            )
        if args.wds_info is None and len(args.data_path) == 1:
            data_path = args.data_path[0]
            if "://" in data_path and args.wds_size is None:
                raise cli.ValidationError("--wds-size is required for remote DATA_PATH")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    evaluate_adversarial_robustness(args)
