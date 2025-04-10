import argparse
import json
import logging
import math
import os
import sys
import time
import typing
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
from birder.transforms.classification import get_rgb_stats
from birder.transforms.detection import batch_images
from birder.transforms.detection import inference_preset
from birder.transforms.detection import training_preset

logger = logging.getLogger(__name__)


def _remove_images_without_annotations(dataset: CocoDetection) -> CocoDetection:
    def _has_only_empty_bbox(anno: list[dict[str, Any]]) -> bool:
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno: list[dict[str, Any]]) -> bool:
        # If it's empty, there is no annotation
        if len(anno) == 0:
            return False

        # If all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    return torch.utils.data.Subset(dataset, ids)


def _convert_to_binary_annotations(dataset: CocoDetection) -> CocoDetection:
    for img_id in dataset.ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        for obj in anno:
            obj["category_id"] = 1

    return dataset


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    #
    # Initialize
    #
    training_utils.init_distributed_mode(args)
    if args.size is None:
        args.size = registry.get_default_size(args.network)

    logger.info(f"Using size={args.size}")

    if args.cpu is True:
        device = torch.device("cpu")
        device_id = 0
    else:
        device = torch.device("cuda")
        device_id = torch.cuda.current_device()

    if args.use_deterministic_algorithms is True:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Enable or disable the autograd anomaly detection
    torch.autograd.set_detect_anomaly(args.grad_anomaly_detection)

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode)
    training_dataset = CocoDetection(
        args.data_path, args.coco_json_path, transforms=training_preset(args.size, args.aug_level, rgb_stats)
    )
    training_dataset = wrap_dataset_for_transforms_v2(training_dataset)
    validation_dataset = CocoDetection(
        args.val_path, args.coco_val_json_path, transforms=inference_preset(args.size, rgb_stats)
    )
    validation_dataset = wrap_dataset_for_transforms_v2(validation_dataset)

    class_to_idx = fs_ops.read_class_file(args.class_file)
    class_to_idx = lib.detection_class_to_idx(class_to_idx)

    if args.binary_mode is True:
        training_dataset = _convert_to_binary_annotations(training_dataset)
        validation_dataset = _convert_to_binary_annotations(validation_dataset)
        class_to_idx = {"Object": 1}

    training_dataset = _remove_images_without_annotations(training_dataset)

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training on {len(training_dataset):,} samples")
    logger.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)  # Does not include background class
    batch_size: int = args.batch_size

    # Data loaders and samplers
    (train_sampler, validation_sampler) = training_utils.get_samplers(args, training_dataset, validation_dataset)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
    )

    last_batch_idx = math.ceil(len(training_dataset) / batch_size) - 1
    begin_epoch = 1
    epochs = args.epochs + 1
    if args.stop_epoch is None:
        args.stop_epoch = epochs
    else:
        args.stop_epoch += 1

    #
    # Initialize network
    #
    sample_shape = (batch_size, args.channels, *args.size)  # B, C, H, W
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
        (net, class_to_idx_saved, training_states) = fs_ops.load_detection_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            config=args.model_config,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_config=args.backbone_model_config,
            backbone_tag=args.backbone_tag,
            epoch=args.resume_epoch,
            new_size=args.size,
        )
        if args.reset_head is True:
            net.reset_classifier(len(class_to_idx))
            net.to(device)
        else:
            assert class_to_idx == class_to_idx_saved

    elif args.pretrained is True:
        (net, class_to_idx_saved, training_states) = fs_ops.load_detection_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            config=args.model_config,
            tag=args.tag,
            backbone=args.backbone,
            backbone_param=args.backbone_param,
            backbone_config=args.backbone_model_config,
            backbone_tag=args.backbone_tag,
            epoch=args.resume_epoch,
            new_size=args.size,
        )
        net.reset_classifier(len(class_to_idx))
        net.to(device)

    else:
        if args.backbone_epoch is not None:
            backbone: DetectorBackbone
            (backbone, class_to_idx_saved, _) = fs_ops.load_checkpoint(
                device,
                args.backbone,
                net_param=args.backbone_param,
                config=args.backbone_model_config,
                tag=args.backbone_tag,
                epoch=args.backbone_epoch,
                new_size=args.size,
            )

        elif args.backbone_pretrained is True:
            (backbone, class_to_idx_saved, _) = fs_ops.load_checkpoint(
                device,
                args.backbone,
                net_param=args.backbone_param,
                config=args.backbone_model_config,
                tag=args.backbone_tag,
                epoch=None,
                new_size=args.size,
            )

        else:
            backbone = registry.net_factory(
                args.backbone,
                sample_shape[1],
                num_outputs,
                net_param=args.backbone_param,
                config=args.backbone_model_config,
                size=args.size,
            )

        net = registry.detection_net_factory(
            args.network,
            num_outputs,
            backbone,
            net_param=args.net_param,
            config=args.model_config,
            size=args.size,
        ).to(device)
        training_states = fs_ops.TrainingStates.empty()

    # Freeze backbone
    if args.freeze_backbone is True:
        net.backbone.freeze()
    elif args.freeze_backbone_stages is not None:
        net.backbone.freeze_stages(up_to_stage=args.freeze_backbone_stages)

    if args.freeze_backbone_bn is True:
        net.backbone = training_utils.freeze_batchnorm2d(net.backbone)

    if args.sync_bn is True and args.distributed is True:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile network
    if args.compile is True:
        net = torch.compile(net)
    elif args.compile_backbone is True:
        net.backbone.detection_features = torch.compile(net.backbone.detection_features)  # type: ignore[method-assign]

    if args.compile_custom is not None:
        mod = getattr(net, args.compile_custom)
        setattr(net, args.compile_custom, torch.compile(mod))

    #
    # Loss criteria, optimizer, learning rate scheduler and training parameter groups
    #

    # Training parameter groups
    custom_keys_weight_decay = training_utils.get_wd_custom_keys(args)
    parameters = training_utils.optimizer_parameter_groups(
        net,
        args.wd,
        norm_weight_decay=args.norm_wd,
        custom_keys_weight_decay=custom_keys_weight_decay,
        layer_decay=args.layer_decay,
        backbone_lr=args.backbone_lr,
    )

    # Learning rate scaling
    lr = training_utils.scale_lr(args)
    grad_accum_steps: int = args.grad_accum_steps

    # Optimizer and learning rate scheduler
    optimizer = training_utils.get_optimizer(parameters, lr, args)
    scheduler = training_utils.get_scheduler(
        args.lr_scheduler,
        optimizer,
        args.warmup_epochs,
        begin_epoch,
        epochs,
        args.lr_cosine_min,
        args.lr_step_size,
        args.lr_steps,
        args.lr_step_gamma,
        args.lr_power,
    )
    if args.compile_opt is True:
        optimizer.step = torch.compile(optimizer.step, fullgraph=False)

    # Gradient scaler and AMP related tasks
    (scaler, amp_dtype) = training_utils.get_amp_scaler(args.amp, args.amp_dtype)

    # Load states
    if args.load_states is True:
        optimizer.load_state_dict(training_states.optimizer_state)
        scheduler.load_state_dict(training_states.scheduler_state)
        if scaler is not None:
            scaler.load_state_dict(training_states.scaler_state)

    elif args.load_scheduler is True:
        scheduler.load_state_dict(training_states.scheduler_state)
        last_lrs = scheduler.get_last_lr()
        for g, last_lr in zip(optimizer.param_groups, last_lrs):
            g["lr"] = last_lr

    last_lr = max(scheduler.get_last_lr())
    if args.plot_lr is True:
        logger.info("Fast forwarding scheduler...")
        lrs = []
        for epoch in range(begin_epoch, epochs):
            optimizer.step()
            lrs.append(max(scheduler.get_last_lr()))
            scheduler.step()

        plt.plot(range(begin_epoch, epochs), lrs)
        plt.show()
        raise SystemExit(0)

    #
    # Distributed (DDP) and Model EMA
    #
    net_without_ddp = net
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        net_without_ddp = net.module

    if args.model_ema is True:
        model_base = net_without_ddp  # Original model without DDP wrapper, will be saved as training state
        model_ema = training_utils.ema_model(args, net_without_ddp, device=device)
        if training_states.ema_model_state is not None:
            logger.info("Setting model EMA weights...")
            if args.compile is True and hasattr(model_ema.module, "_orig_mod") is True:
                model_ema.module._orig_mod.load_state_dict(  # pylint: disable=protected-access
                    training_states.ema_model_state
                )
            else:
                model_ema.module.load_state_dict(training_states.ema_model_state)

            model_ema.n_averaged += 1  # pylint:disable=no-member

        model_to_save = model_ema.module  # Save EMA model weights as default weights
        eval_model = model_ema  # Use EMA for evaluation

    else:
        model_base = None
        model_to_save = net_without_ddp
        eval_model = net

    if args.compile is True and hasattr(model_to_save, "_orig_mod") is True:
        model_to_save = model_to_save._orig_mod  # pylint: disable=protected-access
    if args.compile is True and hasattr(model_base, "_orig_mod") is True:
        model_base = model_base._orig_mod  # type: ignore[union-attr] # pylint: disable=protected-access
    if args.compile_custom is not None and hasattr(getattr(model_to_save, args.compile_custom), "_orig_mod") is True:
        mod = getattr(model_to_save, args.compile_custom)
        setattr(model_to_save, args.compile_custom, mod._orig_mod)  # pylint: disable=protected-access

    #
    # Misc
    #

    # Define metrics
    validation_metrics = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", average="macro").to(device)
    metric_list = ["map", "map_small", "map_medium", "map_large", "map_50", "map_75", "mar_1", "mar_10"]

    # Print network summary
    net_for_info = net_without_ddp
    if args.compile is True and hasattr(net_without_ddp, "_orig_mod") is True:
        net_for_info = net_without_ddp._orig_mod  # pylint: disable=protected-access

    if args.no_summary is False:
        torchinfo.summary(
            net_for_info,
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
    logger.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_detection_signature(input_shape=sample_shape, num_outputs=num_outputs)
    if args.rank == 0:
        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
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

    #
    # Training loop
    #
    logger.info(f"Starting training with learning rate of {last_lr}")
    for epoch in range(begin_epoch, args.stop_epoch):
        tic = time.time()
        net.train()
        running_loss = training_utils.SmoothedValue()
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

        # Zero the parameter gradients
        optimizer.zero_grad()

        for i, (inputs, targets) in enumerate(training_loader):
            inputs = [i.to(device, non_blocking=True) for i in inputs]
            targets = [
                {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
            inputs = batch_images(inputs)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                (_detections, losses) = net(inputs, targets)
                loss = sum(v for v in losses.values())

            if scaler is not None:
                scaler.scale(loss).backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                loss.backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

            # Exponential moving average
            if args.model_ema is True and i % args.model_ema_steps == 0:
                model_ema.update_parameters(net)
                if epoch <= args.warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)  # pylint: disable=no-member

            # Statistics
            running_loss.update(loss.detach())

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                running_loss.synchronize_between_processes(device)
                if args.rank == 0:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )

            # Update progress bar
            if args.rank == 0:
                progress.update(n=batch_size * args.world_size)

        if args.rank == 0:
            progress.close()

        # Epoch training metrics
        epoch_loss = running_loss.global_avg
        logger.info(f"Epoch {epoch}/{epochs-1} training_loss: {epoch_loss:.4f}")

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
                with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                    (detections, losses) = eval_model(inputs)

                for target in targets:
                    # TorchMetrics can't handle "empty" images
                    if "boxes" not in target:
                        target["boxes"] = torch.tensor([], dtype=torch.float, device=device)
                        target["labels"] = torch.tensor([], dtype=torch.int64, device=device)

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
        if last_lr != max(scheduler.get_last_lr()):
            last_lr = max(scheduler.get_last_lr())
            logger.info(f"Updated learning rate to: {last_lr}")

        if args.rank == 0:
            for metric in metric_list:
                summary_writer.add_scalars(
                    "performance", {metric: validation_metrics_dict[metric]}, epoch * len(training_dataset)
                )

            # Epoch validation metrics
            for metric in metric_list:
                logger.info(f"Epoch {epoch}/{epochs-1} {metric}: {validation_metrics_dict[metric]:.3f}")

            # Checkpoint model
            if epoch % args.save_frequency == 0:
                fs_ops.checkpoint_model(
                    network_name,
                    epoch,
                    model_to_save,
                    signature,
                    class_to_idx,
                    rgb_stats,
                    optimizer,
                    scheduler,
                    scaler,
                    model_base,
                )
                if args.keep_last is not None:
                    fs_ops.clean_checkpoints(network_name, args.keep_last)

        # Epoch timing
        toc = time.time()
        (minutes, seconds) = divmod(toc - tic, 60)
        logger.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logger.info("---")

    # Save model hyperparameters with metrics
    if args.rank == 0:
        # Replace list based args
        if args.opt_betas is not None:
            for idx, beta in enumerate(args.opt_betas):
                setattr(args, f"opt_betas_{idx}", beta)

            del args.opt_betas

        if args.lr_steps is not None:
            args.lr_steps = json.dumps(args.lr_steps)
        if args.model_config is not None:
            args.model_config = json.dumps(args.model_config)
        if args.backbone_model_config is not None:
            args.backbone_model_config = json.dumps(args.backbone_model_config)
        if args.size is not None:
            args.size = json.dumps(args.size)

        # Save all args
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
            rgb_stats,
            optimizer,
            scheduler,
            scaler,
            model_base,
        )

    training_utils.shutdown_distributed_mode(args)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train object detection model",
        epilog=(
            "Usage examples\n"
            "==============\n"
            "A classic Faster-RCNN with EfficientNet v2 backbone:\n"
            "torchrun --nproc_per_node=2 train_detection.py  \\\n"
            "    --network faster_rcnn  \\\n"
            "    --backbone efficientnet_v2_s  \\\n"
            "    --backbone-epoch 0  \\\n"
            "    --lr 0.02  \\\n"
            "    --lr-scheduler multistep  \\\n"
            "    --lr-steps 16 22  \\\n"
            "    --lr-step-gamma 0.1  \\\n"
            "    --freeze-backbone-bn  \\\n"
            "    --batch-size 16  \\\n"
            "    --epochs 26  \\\n"
            "    --wd 0.0001  \\\n"
            "    --fast-matmul  \\\n"
            "    --compile-custom backbone_with_fpn\n"
            "\n"
            "A more modern Deformable-DETR example:\n"
            "torchrun --nproc_per_node=2 train_detection.py \\\n"
            "    --network deformable_detr \\\n"
            "    --backbone regnet_y_4g \\\n"
            "    --backbone-epoch 0 \\\n"
            "    --opt adamw \\\n"
            "    --lr 0.0002 \\\n"
            "    --backbone-lr 0.00002 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --freeze-backbone-bn \\\n"
            "    --batch-size 8 \\\n"
            "    --epochs 50 \\\n"
            "    --wd 0.0001 \\\n"
            "    --clip-grad-norm 1 \\\n"
            "    --fast-matmul \\\n"
            "    --compile-backbone \\\n"
            "    --compile-opt\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the neural network to use")
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required by some networks")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--backbone", type=str, help="the neural network to used as backbone")
    parser.add_argument(
        "--backbone-param",
        type=float,
        help="network specific parameter, required by some networks (for the backbone)",
    )
    parser.add_argument(
        "--backbone-model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the backbone default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    parser.add_argument("--backbone-epoch", type=int, help="load backbone weights from selected epoch")
    parser.add_argument(
        "--backbone-pretrained",
        default=False,
        action="store_true",
        help="start with pretrained version of specified backbone",
    )
    parser.add_argument(
        "--pretrained",
        default=False,
        action="store_true",
        help="start with pretrained version of specified network, reset the classification head",
    )
    parser.add_argument("--reset-head", default=False, action="store_true", help="reset the classification head")
    parser.add_argument("--freeze-backbone", default=False, action="store_true", help="freeze backbone")
    parser.add_argument("--freeze-backbone-stages", type=int, help="number of backbone stages to freeze")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-backbone", default=False, action="store_true", help="enable backbone only compilation"
    )
    parser.add_argument("--compile-custom", type=str, help="enable compilation for specific module")
    parser.add_argument(
        "--compile-opt", default=False, action="store_true", help="enable compilation for optimizer step"
    )
    training_utils.add_optimizer_args(parser)
    parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
    parser.add_argument(
        "--lr-scale", type=int, help="reference batch size for LR scaling, if provided, LR will be scaled accordingly"
    )
    parser.add_argument(
        "--lr-scale-type", type=str, choices=["linear", "sqrt"], default="linear", help="learning rate scaling type"
    )
    parser.add_argument("--backbone-lr", type=float, help="backbone learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--norm-wd", type=float, help="weight decay for Normalization layers")
    parser.add_argument("--bias-weight-decay", type=float, help="weight decay for bias parameters of all layers")
    parser.add_argument(
        "--transformer-embedding-decay",
        type=float,
        help="weight decay for embedding parameters for vision transformer models",
    )
    parser.add_argument("--layer-decay", type=float, help="layer-wise learning rate decay (LLRD)")
    training_utils.add_scheduler_args(parser)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, metavar="N", help="number of steps to accumulate gradients"
    )
    parser.add_argument("--channels", type=int, default=3, metavar="N", help="no. of image channels")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size (defaults to network recommendation)"
    )
    parser.add_argument(
        "--freeze-backbone-bn",
        default=False,
        action="store_true",
        help="freeze all batch statistics and affine parameters of batchnorm2d layers (backbone only)",
    )
    parser.add_argument("--sync-bn", default=False, action="store_true", help="use synchronized BatchNorm")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N", help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N", help="number of warmup epochs")
    parser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="magnitude of augmentations (0 off -> 3 highest)",
    )
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of training epochs")
    parser.add_argument(
        "--stop-epoch", type=int, metavar="N", help="epoch to stop the training at (multi step training)"
    )
    parser.add_argument("--save-frequency", type=int, default=5, metavar="N", help="frequency of model saving")
    parser.add_argument("--keep-last", type=int, metavar="N", help="number of checkpoints to keep")
    parser.add_argument("--resume-epoch", type=int, metavar="N", help="epoch to resume training from")
    parser.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
    parser.add_argument("--load-scheduler", default=False, action="store_true", help="load scheduler only resuming")
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
    parser.add_argument(
        "--log-interval", type=int, default=20, metavar="N", help="how many steps between summary writes"
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=min(16, max(os.cpu_count() // 4, 4)),  # type: ignore[operator]
        metavar="N",
        help="number of preprocessing workers",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
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
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument(
        "--grad-anomaly-detection",
        default=False,
        action="store_true",
        help="enable the autograd anomaly detection (for debugging)",
    )
    parser.add_argument("--world-size", type=int, default=1, metavar="N", help="number of distributed processes")
    parser.add_argument("--dist-url", type=str, default="env://", help="url used to set up distributed training")
    parser.add_argument("--clip-grad-norm", type=float, help="the maximum gradient norm")
    parser.add_argument("--gpu", type=int, metavar="ID", help="gpu id to use (ignored in distributed mode)")
    parser.add_argument("--cpu", default=False, action="store_true", help="use cpu (mostly for testing)")
    parser.add_argument(
        "--use-deterministic-algorithms", default=False, action="store_true", help="use only deterministic algorithms"
    )
    parser.add_argument(
        "--binary-mode",
        default=False,
        action="store_true",
        help="treat all objects as a single class (binary detection: object vs background)",
    )
    parser.add_argument(
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
    parser.add_argument("--no-summary", default=False, action="store_true", help="don't print model summary")
    parser.add_argument(
        "--val-path", type=str, default=str(settings.DETECTION_DATA_PATH), help="validation base directory path"
    )
    parser.add_argument(
        "--data-path", type=str, default=str(settings.DETECTION_DATA_PATH), help="training base directory path"
    )
    parser.add_argument(
        "--coco-val-json-path",
        type=str,
        default=f"{settings.VALIDATION_DETECTION_ANNOTATIONS_PATH}_coco.json",
        help="validation COCO json path",
    )
    parser.add_argument(
        "--coco-json-path",
        type=str,
        default=f"{settings.TRAINING_DETECTION_ANNOTATIONS_PATH}_coco.json",
        help="training COCO json path",
    )
    parser.add_argument(
        "--class-file",
        type=str,
        default=str(settings.DETECTION_DATA_PATH.joinpath(settings.CLASS_LIST_NAME)),
        metavar="FILE",
        help="class list file",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = str(args.data_path)
    args.val_path = str(args.val_path)
    assert args.network is not None
    assert args.backbone is not None
    assert (
        args.backbone_pretrained is False or args.backbone_epoch is None
    ), "Cannot set backbone epoch while starting from a pretrained backbone"
    assert (
        args.pretrained is False or args.resume_epoch is None
    ), "Cannot set resume epoch while starting from a pretrained network"
    assert args.load_states is False or (
        args.load_states is True and args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert (
        args.load_scheduler is False or args.resume_epoch is not None
    ), "Load scheduler must be from resumed training (--resume-epoch)"
    assert args.freeze_backbone is False or args.freeze_backbone_stages is None
    assert (
        registry.exists(args.network, task=Task.OBJECT_DETECTION) is True
    ), "Unknown network, see list-models tool for available options"
    assert (
        registry.exists(args.backbone, net_type=DetectorBackbone) is True
    ), "Unknown backbone, see list-models tool for available options"
    assert args.compile is False or args.compile_backbone is False
    args.size = cli.parse_size(args.size)


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
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    train(args)


if __name__ == "__main__":
    logger = logging.getLogger(__spec__.name)
    main()
