import argparse
import json
import logging
import math
import sys
import time
import typing
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.amp
import torch.utils.data
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common import training_utils
from birder.conf import settings
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.detection.base import get_detection_signature
from birder.transforms.classification import RGBMode
from birder.transforms.classification import get_rgb_values
from birder.transforms.detection import batch_images
from birder.transforms.detection import inference_preset
from birder.transforms.detection import training_preset


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    training_utils.init_distributed_mode(args)
    if args.size is None:
        args.size = registry.get_default_size(args.network)

    logging.info(f"Using size={args.size}")

    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    torch.backends.cudnn.benchmark = True

    rgb_values = get_rgb_values(args.rgb_mode)
    train_base_name = Path(args.data_path).stem
    train_coco_path = Path(args.data_path).parent.joinpath(f"{train_base_name}_coco.json")
    val_base_name = Path(args.val_path).stem
    val_coco_path = Path(args.val_path).parent.joinpath(f"{val_base_name}_coco.json")

    training_dataset = CocoDetection(
        ".", train_coco_path, transforms=training_preset(args.size, args.aug_level, rgb_values)
    )
    training_dataset = wrap_dataset_for_transforms_v2(training_dataset)
    validation_dataset = CocoDetection(".", val_coco_path, transforms=inference_preset(args.size, rgb_values))
    validation_dataset = wrap_dataset_for_transforms_v2(validation_dataset)
    class_to_idx = fs_ops.read_class_file(settings.DETECTION_DATA_PATH.joinpath(settings.CLASS_LIST_NAME))
    class_to_idx = lib.detection_class_to_idx(class_to_idx)

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logging.info(f"Using device {device}:{device_id}")
    logging.info(f"Training on {len(training_dataset):,} samples")
    logging.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)  # Does not include background class
    batch_size: int = args.batch_size
    begin_epoch = 1
    epochs = args.epochs + 1
    if args.stop_epoch is None:
        args.stop_epoch = epochs
    else:
        args.stop_epoch += 1

    # Initialize network
    sample_shape = (batch_size,) + (args.channels, args.size, args.size)  # B, C, H, W
    network_name = lib.get_detection_network_name(
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        backbone=args.backbone,
        backbone_param=args.backbone_param,
        backbone_tag=args.backbone_tag,
    )

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, class_to_idx_saved, optimizer_state, scheduler_state, scaler_state) = fs_ops.load_detection_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_tag=args.backbone_tag,
            epoch=args.resume_epoch,
        )
        assert class_to_idx == class_to_idx_saved

    else:
        if args.backbone_epoch is not None:
            backbone: DetectorBackbone
            (backbone, class_to_idx_saved, _, _, _) = fs_ops.load_checkpoint(
                device,
                args.backbone,
                net_param=args.backbone_param,
                tag=args.backbone_tag,
                epoch=args.backbone_epoch,
                new_size=args.size,
            )

        else:
            backbone = registry.net_factory(args.backbone, sample_shape[1], num_outputs, args.backbone_param, args.size)

        net = registry.detection_net_factory(args.network, num_outputs, backbone, args.net_param, args.size).to(device)

    # Freeze backbone
    if args.freeze_backbone is True:
        net.backbone.freeze()
    elif args.freeze_backbone_stages is not None:
        net.backbone.freeze_stages(up_to_stage=args.freeze_backbone_stages)

    if args.fast_matmul is True:
        torch.set_float32_matmul_precision("high")

    # Compile network
    if args.compile is True:
        raise NotImplementedError

    # Define optimizer and learning rate scheduler and training parameter groups
    custom_keys_weight_decay = training_utils.get_wd_custom_keys(args)
    parameters = training_utils.optimizer_parameter_groups(
        net,
        args.wd,
        norm_weight_decay=args.norm_wd,
        custom_keys_weight_decay=custom_keys_weight_decay,
        layer_decay=args.layer_decay,
    )
    optimizer = training_utils.get_optimizer(args.opt, parameters, args.lr, args.wd, args.momentum, args.nesterov)
    scheduler = training_utils.get_scheduler(
        args.lr_scheduler,
        optimizer,
        args.warmup_epochs,
        begin_epoch,
        epochs,
        args.lr_cosine_min,
        args.lr_step_size,
        args.lr_step_gamma,
    )

    # Gradient scaler and AMP related tasks
    (scaler, amp_dtype) = training_utils.get_amp_scaler(args.amp, args.amp_dtype)

    # Load states
    if args.load_states is True:
        optimizer.load_state_dict(optimizer_state)  # pylint: disable=possibly-used-before-assignment
        scheduler.load_state_dict(scheduler_state)  # pylint: disable=possibly-used-before-assignment
        if scaler is not None:
            scaler.load_state_dict(scaler_state)  # pylint: disable=possibly-used-before-assignment

    last_lr = scheduler.get_last_lr()[0]
    if args.plot_lr is True:
        logging.info("Fast forwarding scheduler...")
        lrs = []
        for epoch in range(begin_epoch, epochs):
            optimizer.step()
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        plt.plot(range(begin_epoch, epochs), lrs)
        plt.show()
        raise SystemExit(0)

    # Distributed
    net_without_ddp = net
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        net_without_ddp = net.module

    # Model EMA
    if args.model_ema is True:
        model_ema = training_utils.ema_model(args, net_without_ddp, device=device)
        model_to_save = model_ema.module
        eval_model = model_ema

    else:
        model_to_save = net_without_ddp
        eval_model = net

    # Define metrics
    validation_metrics = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", average="macro").to(device)
    metric_list = ["map", "map_small", "map_medium", "map_large", "map_50", "map_75", "mar_1", "mar_10"]

    # Print network summary
    torchinfo.summary(
        net_without_ddp,
        device=device,
        input_size=sample_shape,
        dtypes=[torch.float32],
        col_names=["input_size", "output_size", "kernel_size", "num_params"],
        depth=4,
        verbose=1 if args.rank == 0 else 0,
    )

    # Training logs
    training_log_name = training_utils.training_log_name(network_name, device)
    training_log_path = settings.TRAINING_RUNS_PATH.joinpath(training_log_name)
    logging.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_detection_signature(input_shape=sample_shape, num_outputs=num_outputs)
    if args.rank == 0:
        summary_writer.flush()
        fs_ops.write_signature(network_name, signature)
        with open(training_log_path.joinpath("args.json"), "w", encoding="utf-8") as handle:
            json.dump({"cmdline": " ".join(sys.argv), **vars(args)}, handle, indent=2)

        with open(training_log_path.joinpath("training_data.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "training_samples": len(training_dataset),
                    "validation_samples": len(validation_dataset),
                    "classes": list(class_to_idx.keys()),
                },
                handle,
                indent=2,
            )

    # Data loaders and samplers
    (train_sampler, validation_sampler) = training_utils.get_samplers(args, training_dataset, validation_dataset)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
    )

    last_batch_idx = math.ceil(len(training_dataset) / batch_size) - 1

    # Enable or disable the autograd anomaly detection
    torch.autograd.set_detect_anomaly(args.grad_anomaly_detection)

    # Training loop
    logging.info(f"Starting training with learning rate of {last_lr}")
    for epoch in range(begin_epoch, args.stop_epoch):
        tic = time.time()
        net.train()
        running_loss = 0.0
        validation_metrics.reset()

        if args.distributed is True:
            train_sampler.set_epoch(epoch)

        if args.rank == 0:
            progress = tqdm(
                desc=f"Epoch {epoch}/{epochs-1}",
                total=len(training_dataset),
                initial=0,
                unit="samples",
                leave=False,
            )

        for i, (inputs, targets) in enumerate(training_loader):
            inputs = [i.to(device, non_blocking=True) for i in inputs]
            targets = [
                {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
            inputs = batch_images(inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                (_detections, losses) = net(inputs, targets)
                loss = sum(v for v in losses.values())

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                optimizer.step()

            # Exponential moving average
            if args.model_ema is True and i % args.model_ema_steps == 0:
                model_ema.update_parameters(net)
                if epoch < args.warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)  # pylint: disable=no-member

            # Statistics
            running_loss += loss.item() * inputs.size(0)

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                summary_writer.add_scalars(
                    "loss",
                    {f"training{args.rank}": running_loss / (i * batch_size)},
                    ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                )

            # Update progress bar
            if args.rank == 0:
                progress.update(n=batch_size * args.world_size)

        if args.rank == 0:
            progress.close()

        epoch_loss = running_loss / len(training_dataset)

        # Epoch training metrics
        epoch_loss = training_utils.reduce_across_processes(epoch_loss, device)
        logging.info(f"Epoch {epoch}/{epochs-1} training_loss: {epoch_loss:.4f}")

        # Validation
        eval_model.eval()
        if args.rank == 0:
            progress = tqdm(
                desc=f"Epoch {epoch}/{epochs-1}",
                total=len(validation_dataset),
                initial=0,
                unit="samples",
                leave=False,
            )

        with torch.inference_mode():
            for inputs, targets in validation_loader:
                inputs = [i.to(device, non_blocking=True) for i in inputs]
                targets = [
                    {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]
                inputs = batch_images(inputs)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    (detections, losses) = eval_model(inputs)

                # Statistics
                validation_metrics(detections, targets)

                # Update progress bar
                if args.rank == 0:
                    progress.update(n=batch_size * args.world_size)

        if args.rank == 0:
            progress.close()

        validation_metrics_dict = validation_metrics.compute()

        # Learning rate scheduler update
        scheduler.step()
        if last_lr != scheduler.get_last_lr()[0]:
            last_lr = scheduler.get_last_lr()[0]
            logging.info(f"Updated learning rate to: {last_lr}")

        if args.rank == 0:
            for metric in metric_list:
                summary_writer.add_scalars(
                    "performance", {metric: validation_metrics_dict[metric]}, epoch * len(training_dataset)
                )

            # Epoch validation metrics
            for metric in metric_list:
                logging.info(f"Epoch {epoch}/{epochs-1} {metric}: {validation_metrics_dict[metric]:.3f}")

            # Checkpoint model
            if epoch % args.save_frequency == 0:
                fs_ops.checkpoint_model(
                    network_name,
                    epoch,
                    model_to_save,
                    signature,
                    class_to_idx,
                    rgb_values,
                    optimizer,
                    scheduler,
                    scaler,
                )

        # Epoch timing
        toc = time.time()
        (minutes, seconds) = divmod(toc - tic, 60)
        logging.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logging.info("---")

    # Save model hyperparameters with metrics
    if args.rank == 0:
        val_metrics = validation_metrics.compute()
        summary_writer.add_hparams(
            {**vars(args), "training_samples": len(training_dataset)},
            {"hparam/val_map": val_metrics["map"]},
        )

    summary_writer.close()

    # Checkpoint model
    if args.distributed is False or (args.distributed is True and args.rank == 0):
        fs_ops.checkpoint_model(
            network_name,
            epoch,
            model_to_save,
            signature,
            class_to_idx,
            rgb_values,
            optimizer,
            scheduler,
            scaler,
        )


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train object detection model",
        epilog=(
            "Usage examples:\n"
            "python train_detection.py --network faster_rcnn --backbone mobilenet_v3_large --backbone-param 1 "
            "--backbone-epoch 0 --lr 0.02 --batch-size 16\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the neural network to use")
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required for most networks")
    parser.add_argument("--backbone", type=str, help="the neural network to used as backbone")
    parser.add_argument(
        "--backbone-param",
        type=float,
        help="network specific parameter, required by most networks (for the backbone)",
    )
    parser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    parser.add_argument("--backbone-epoch", type=int, help="load backbone weights from selected epoch")
    parser.add_argument("--freeze-backbone", default=False, action="store_true", help="freeze backbone")
    parser.add_argument("--freeze-backbone-stages", type=int, help="number of backbone stages to freeze")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--opt",
        type=str,
        choices=list(typing.get_args(training_utils.OptimizerType)),
        default="sgd",
        help="optimizer to use",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer momentum")
    parser.add_argument("--nesterov", default=False, action="store_true", help="use nesterov momentum")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--norm-wd", type=float, default=None, help="weight decay for Normalization layers")
    parser.add_argument(
        "--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers"
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models",
    )
    parser.add_argument("--layer-decay", type=float, default=None, help="layer-wise learning rate decay (LLRD)")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=list(typing.get_args(training_utils.SchedulerType)),
        default="constant",
        help="learning rate scheduler",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=40,
        help="decrease lr every step-size epochs (for step scheduler only)",
    )
    parser.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.75,
        help="multiplicative factor of learning rate decay (for step scheduler only)",
    )
    parser.add_argument(
        "--lr-cosine-min",
        type=float,
        default=0.000001,
        help="minimum learning rate (for cosine annealing scheduler only)",
    )
    parser.add_argument("--channels", type=int, default=3, help="no. of image channels")
    parser.add_argument("--size", type=int, default=None, help="image size (defaults to network recommendation)")
    parser.add_argument("--batch-size", type=int, default=16, help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="number of warmup epochs")
    parser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="magnitude of augmentations (0 off -> 2 highest)",
    )
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="calculated",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument(
        "--stop-epoch", type=int, default=None, help="epoch to stop the training at (multi step training)"
    )
    parser.add_argument("--save-frequency", type=int, default=5, help="frequency of model saving")
    parser.add_argument("--resume-epoch", type=int, help="epoch to resume training from")
    parser.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
    parser.add_argument(
        "--model-ema",
        default=False,
        action="store_true",
        help="enable tracking exponential moving average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.9999,
        help="decay factor for exponential moving average of model parameters",
    )
    parser.add_argument(
        "--ra-sampler",
        default=False,
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument("--ra-reps", type=int, default=3, help="number of repetitions for Repeated Augmentation")
    parser.add_argument("-t", "--tag", type=str, help="add training logs tag")
    parser.add_argument("--log-interval", type=int, default=20, help="how many steps between summary writes")
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=8,
        help="number of preprocessing workers",
    )
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.cuda.amp for mixed precision training"
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    parser.add_argument(
        "--fast-matmul",
        default=False,
        action="store_true",
        help="use fast matrix multiplication (affects precision)",
    )
    parser.add_argument(
        "--grad-anomaly-detection",
        default=False,
        action="store_true",
        help="enable the autograd anomaly detection (for debugging)",
    )
    parser.add_argument("--world-size", type=int, default=1, help="number of distributed processes")
    parser.add_argument("--dist-url", type=str, default="env://", help="url used to set up distributed training")
    parser.add_argument("--clip-grad-norm", type=float, default=None, help="the maximum gradient norm")
    parser.add_argument("--gpu", type=int, help="gpu id to use (ignored in distributed mode)")
    parser.add_argument(
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=str(settings.VALIDATION_DETECTION_ANNOTATIONS_PATH),
        help="validation directory path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.TRAINING_DETECTION_ANNOTATIONS_PATH),
        help="training directory path",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.network is not None
    assert args.backbone is not None
    assert args.load_states is False or (
        args.load_states is True and args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert args.freeze_backbone is False or args.freeze_backbone_stages is None
    assert (
        registry.exists(args.network, task=Task.OBJECT_DETECTION) is True
    ), "Unknown network, see list-models tool for available options"
    assert (
        registry.exists(args.backbone, net_type=DetectorBackbone) is True
    ), "Unknown backbone, see list-models tool for available options"


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    args = argparse.Namespace(**kwargs)
    args = parser.parse_args([], args)
    validate_args(args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    if settings.MODELS_DIR.exists() is False:
        logging.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    train(args)


if __name__ == "__main__":
    main()
