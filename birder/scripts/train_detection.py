import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

import birder
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common import training_cli
from birder.common import training_utils
from birder.conf import settings
from birder.data.collators.detection import training_collate_fn
from birder.data.datasets.coco import CocoMosaicTraining
from birder.data.datasets.coco import CocoTraining
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.detection import InferenceTransform
from birder.data.transforms.detection import training_preset
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.detection.base import get_detection_signature

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    #
    # Initialize
    #
    training_utils.init_distributed_mode(args)
    logger.info(f"Starting training, birder version: {birder.__version__}, pytorch version: {torch.__version__}")
    training_utils.log_git_info()

    if (
        args.multiscale is True
        or args.dynamic_size is True
        or args.max_size is not None
        or args.aug_type == "multiscale"
        or args.aug_type == "detr"
    ):
        dynamic_size = True
    else:
        dynamic_size = False

    if args.size is None:
        args.size = registry.get_default_size(args.network)

    if dynamic_size is False:
        logger.info(f"Using size={args.size}")
    else:
        logger.info(f"Running with dynamic size, with base size={args.size}")

    if args.cpu is True:
        device = torch.device("cpu")
        device_id = 0
    else:
        device = torch.device("cuda")
        device_id = torch.cuda.current_device()

    if args.use_deterministic_algorithms is True:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    elif dynamic_size is False:
        torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        lib.set_random_seeds(args.seed)

    if args.non_interactive is True or training_utils.is_local_primary(args) is False:
        disable_tqdm = True
    elif sys.stderr.isatty() is False:
        disable_tqdm = True
    else:
        disable_tqdm = False

    # Enable or disable the autograd anomaly detection
    torch.autograd.set_detect_anomaly(args.grad_anomaly_detection)

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    logger.debug(f"Using RGB stats: {rgb_stats}")

    transforms = training_preset(
        args.size, args.aug_type, args.aug_level, rgb_stats, args.dynamic_size, args.multiscale, args.max_size
    )
    mosaic_transforms = training_preset(
        args.size,
        args.aug_type,
        args.aug_level,
        rgb_stats,
        args.dynamic_size,
        args.multiscale,
        args.max_size,
        post_mosaic=True,
    )
    if args.mosaic_prob > 0.0:
        if args.dynamic_size is True or args.multiscale is True:
            # Dynamic/Multiscale: args.size is the short-side target
            if args.max_size is not None:
                mosaic_dim = args.max_size
            else:
                mosaic_dim = min(args.size) * 2

        else:
            # Fixed size
            mosaic_dim = max(args.size) * 2

        training_dataset = CocoMosaicTraining(
            args.data_path,
            args.coco_json_path,
            transforms=transforms,
            mosaic_transforms=mosaic_transforms,
            output_size=(mosaic_dim, mosaic_dim),
            fill_value=114,
            mosaic_prob=args.mosaic_prob,
            mosaic_type=args.mosaic_type,
        )
    else:
        training_dataset = CocoTraining(args.data_path, args.coco_json_path, transforms=transforms)

    validation_dataset = CocoTraining(
        args.val_path,
        args.coco_val_json_path,
        transforms=InferenceTransform(args.size, rgb_stats, dynamic_size, args.max_size),
    )

    if args.class_file is not None:
        class_to_idx = fs_ops.read_class_file(args.class_file)
    else:
        class_to_idx = lib.class_to_idx_from_coco(training_dataset.dataset.coco.cats)

    class_to_idx = lib.detection_class_to_idx(class_to_idx)
    if args.ignore_file is not None:
        with open(args.ignore_file, "r", encoding="utf-8") as handle:
            ignore_list = handle.read().splitlines()
    else:
        ignore_list = []

    if args.binary_mode is True:
        training_dataset.convert_to_binary_annotations()
        validation_dataset.convert_to_binary_annotations()
        class_to_idx = training_dataset.class_to_idx

    training_dataset.remove_images_without_annotations(ignore_list)

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training on {len(training_dataset):,} samples")
    logger.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)  # Does not include background class
    batch_size: int = args.batch_size
    model_ema_steps: int = args.model_ema_steps * args.grad_accum_steps
    logger.debug(f"Effective batch size = {args.batch_size * args.grad_accum_steps * args.world_size}")

    # Data loaders and samplers
    (train_sampler, validation_sampler) = training_utils.get_samplers(args, training_dataset, validation_dataset)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=training_collate_fn,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=training_collate_fn,
        pin_memory=True,
        drop_last=args.drop_last,
    )

    last_batch_idx = len(training_loader) - 1
    begin_epoch = 1
    epochs = args.epochs + 1
    if args.stop_epoch is None:
        args.stop_epoch = epochs
    else:
        args.stop_epoch += 1

    #
    # Initialize network
    #
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    sample_shape = (batch_size, args.channels, *args.size)  # B, C, H, W
    network_name = lib.get_detection_network_name(
        args.network, tag=args.tag, backbone=args.backbone, backbone_tag=args.backbone_tag
    )

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, class_to_idx_saved, training_states) = fs_ops.load_detection_checkpoint(
            device,
            args.network,
            config=args.model_config,
            tag=args.tag,
            backbone=args.backbone,
            backbone_config=args.backbone_model_config,
            backbone_tag=args.backbone_tag,
            epoch=args.resume_epoch,
            new_size=args.size,
            strict=not args.non_strict_weights,
        )
        if args.reset_head is True:
            net.reset_classifier(len(class_to_idx))
        else:
            assert class_to_idx == class_to_idx_saved

    elif args.pretrained is True:
        fs_ops.download_model_by_weights(network_name, progress_bar=training_utils.is_local_primary(args))
        (net, class_to_idx_saved, training_states) = fs_ops.load_detection_checkpoint(
            device,
            args.network,
            config=args.model_config,
            tag=args.tag,
            backbone=args.backbone,
            backbone_config=args.backbone_model_config,
            backbone_tag=args.backbone_tag,
            epoch=None,
            new_size=args.size,
            strict=not args.non_strict_weights,
        )
        if args.reset_head is True:
            net.reset_classifier(len(class_to_idx))
        else:
            assert class_to_idx == class_to_idx_saved

    else:
        if args.backbone_epoch is not None:
            backbone: DetectorBackbone
            (backbone, class_to_idx_saved, _) = fs_ops.load_checkpoint(
                device,
                args.backbone,
                config=args.backbone_model_config,
                tag=args.backbone_tag,
                epoch=args.backbone_epoch,
                new_size=args.size,
                strict=not args.non_strict_weights,
            )

        elif args.backbone_pretrained is True:
            fs_ops.download_model_by_weights(
                lib.get_network_name(args.backbone, tag=args.backbone_tag),
                progress_bar=training_utils.is_local_primary(args),
            )
            (backbone, class_to_idx_saved, _) = fs_ops.load_checkpoint(
                device,
                args.backbone,
                config=args.backbone_model_config,
                tag=args.backbone_tag,
                epoch=None,
                new_size=args.size,
                strict=not args.non_strict_weights,
            )

        else:
            backbone = registry.net_factory(
                args.backbone, sample_shape[1], num_outputs, config=args.backbone_model_config, size=args.size
            )

        net = registry.detection_net_factory(
            args.network, num_outputs, backbone, config=args.model_config, size=args.size
        )
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
    if dynamic_size is True:
        net.set_dynamic_size()

    # Freeze
    if args.freeze_body is True:
        net.freeze(freeze_classifier=False)
    elif args.freeze_backbone is True:
        net.backbone.freeze()
    elif args.freeze_backbone_stages is not None:
        net.backbone.freeze_stages(up_to_stage=args.freeze_backbone_stages)

    if args.freeze_bn is True:
        net = training_utils.freeze_batchnorm2d(net)
    elif args.freeze_backbone_bn is True:
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
        layer_decay_min_scale=args.layer_decay_min_scale,
        layer_decay_no_opt_scale=args.layer_decay_no_opt_scale,
        bias_lr=args.bias_lr,
        backbone_lr=args.backbone_lr,
    )

    # Learning rate scaling
    lr = training_utils.scale_lr(args)
    grad_accum_steps: int = args.grad_accum_steps

    if args.lr_scheduler_update == "epoch":
        iter_update = False
        iters_per_epoch = 1
    elif args.lr_scheduler_update == "iter":
        iter_update = True
        iters_per_epoch = math.ceil(len(training_loader) / grad_accum_steps)
    else:
        raise ValueError("Unsupported lr_scheduler_update")

    # Optimizer and learning rate scheduler
    optimizer = training_utils.get_optimizer(parameters, lr, args)
    scheduler = training_utils.get_scheduler(optimizer, iters_per_epoch, args)
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

    last_lr = float(max(scheduler.get_last_lr()))
    if args.plot_lr is True:
        logger.info("Fast forwarding scheduler...")
        optimizer.step()
        lrs = []
        for _ in range(begin_epoch, epochs):
            for _ in range(iters_per_epoch):
                lrs.append(float(max(scheduler.get_last_lr())))
                scheduler.step()

        plt.plot(np.linspace(begin_epoch, epochs, iters_per_epoch * (epochs - begin_epoch), endpoint=False), lrs)
        plt.show()
        raise SystemExit(0)

    #
    # Distributed (DDP) and Model EMA
    #
    if args.model_ema_warmup is not None:
        ema_warmup_epochs = args.model_ema_warmup
    elif args.warmup_epochs is not None:
        ema_warmup_epochs = args.warmup_epochs
    elif args.warmup_iters is not None:
        ema_warmup_epochs = args.warmup_iters // iters_per_epoch
    else:
        ema_warmup_epochs = 0

    logger.debug(f"EMA warmup epochs = {ema_warmup_epochs}")
    net_without_ddp = net
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters
        )
        net_without_ddp = net.module

    if args.model_ema is True:
        model_base = net_without_ddp  # Original model without DDP wrapper, will be saved as training state
        model_ema = training_utils.ema_model(args, net_without_ddp, device=device)
        if args.load_states is True and training_states.ema_model_state is not None:
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
        summary = torchinfo.summary(
            net_for_info,
            device=device,
            input_size=sample_shape,
            dtypes=[model_dtype],
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
            verbose=0,
        )
        if training_utils.is_global_primary(args) is True:
            # Write to stderr, same as all the logs
            print(summary, file=sys.stderr)

    # Training logs
    training_log_path = training_utils.training_log_path(network_name, device, args.experiment)
    logger.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_detection_signature(input_shape=sample_shape, num_outputs=num_outputs, dynamic=dynamic_size)
    file_handler: logging.Handler = logging.NullHandler()
    if training_utils.is_local_primary(args) is True:
        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        file_handler = training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
        with open(training_log_path.joinpath("training_args.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "birder_version": birder.__version__,
                    "pytorch_version": torch.__version__,
                    "cmdline": " ".join(sys.argv),
                    **vars(args),
                },
                handle,
                indent=2,
            )

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

        progress = tqdm(
            desc=f"Epoch {epoch}/{epochs-1}",
            total=len(training_dataset),
            leave=False,
            disable=disable_tqdm,
            unit="samples",
            initial=0,
        )

        # Zero the parameter gradients
        optimizer.zero_grad()

        epoch_start = time.time()
        start_time = epoch_start
        last_idx = 0
        for i, (inputs, targets, masks, image_sizes) in enumerate(training_loader):
            inputs = inputs.to(device, dtype=model_dtype, non_blocking=True)
            targets = [
                {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
            masks = masks.to(device, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                (_detections, losses) = net(inputs, targets, masks, image_sizes)
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
                    if iter_update is True:
                        scheduler.step()

            else:
                loss.backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                    if iter_update is True:
                        scheduler.step()

            # Exponential moving average
            if args.model_ema is True and i % model_ema_steps == 0:
                model_ema.update_parameters(net)
                if epoch <= ema_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)  # pylint: disable=no-member

            # Statistics
            running_loss.update(loss.detach())

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                time_now = time.time()
                time_cost = time_now - start_time
                steps_processed_in_interval = i - last_idx
                rate = steps_processed_in_interval * (batch_size * args.world_size) / time_cost

                avg_time_per_step = time_cost / steps_processed_in_interval
                remaining_steps_in_epoch = last_batch_idx - i
                estimated_time_to_finish_epoch = remaining_steps_in_epoch * avg_time_per_step

                start_time = time_now
                last_idx = i
                cur_lr = float(max(scheduler.get_last_lr()))

                running_loss.synchronize_between_processes(device)
                with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                    log.info(
                        f"[Trn] Epoch {epoch}/{epochs-1}, step {i+1}/{last_batch_idx+1}  "
                        f"Loss: {running_loss.avg:.4f}  "
                        f"Elapsed: {lib.format_duration(time_now-epoch_start)}  "
                        f"ETA: {lib.format_duration(estimated_time_to_finish_epoch)}  "
                        f"T: {time_cost:.1f}s  "
                        f"R: {rate:.1f} samples/s  "
                        f"LR: {cur_lr:.4e}"
                    )

                if training_utils.is_local_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )

            # Update progress bar
            progress.update(n=batch_size * args.world_size)

        progress.close()

        # Epoch training metrics
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} training_loss: {running_loss.global_avg:.4f}")

        # Validation
        eval_model.eval()
        progress = tqdm(
            desc=f"Epoch {epoch}/{epochs-1}",
            total=len(validation_dataset),
            leave=False,
            disable=disable_tqdm,
            unit="samples",
            initial=0,
        )
        with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
            log.info(f"[Val] Starting validation for epoch {epoch}/{epochs-1}...")

        epoch_start = time.time()
        with torch.inference_mode():
            for inputs, targets, masks, image_sizes in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = [
                    {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                    (detections, losses) = eval_model(inputs, masks=masks, image_sizes=image_sizes)

                for target in targets:
                    # TorchMetrics can't handle "empty" images
                    if "boxes" not in target:
                        target["boxes"] = torch.tensor([], dtype=torch.float, device=device)
                        target["labels"] = torch.tensor([], dtype=torch.int64, device=device)

                # Statistics
                validation_metrics(detections, targets)

                # Update progress bar
                if training_utils.is_local_primary(args) is True:
                    progress.update(n=batch_size * args.world_size)

        time_now = time.time()
        rate = len(validation_dataset) / (time_now - epoch_start)
        with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
            log.info(
                f"[Val] Epoch {epoch}/{epochs-1} "
                f"Elapsed: {lib.format_duration(time_now-epoch_start)}  "
                f"R: {rate:.1f} samples/s"
            )

        progress.close()

        validation_metrics_dict = validation_metrics.compute()

        # Write statistics
        if training_utils.is_local_primary(args) is True:
            for metric in metric_list:
                summary_writer.add_scalars(
                    "performance", {metric: validation_metrics_dict[metric]}, epoch * len(training_dataset)
                )

        # Epoch validation metrics
        for metric in metric_list:
            logger.info(f"[Val] Epoch {epoch}/{epochs-1} {metric}: {validation_metrics_dict[metric]:.4f}")

        # Learning rate scheduler update
        if iter_update is False:
            scheduler.step()
        if last_lr != float(max(scheduler.get_last_lr())):
            last_lr = float(max(scheduler.get_last_lr()))
            logger.info(f"Updated learning rate to: {last_lr}")

        if training_utils.is_local_primary(args) is True:
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
        logger.info(f"Total time: {lib.format_duration(toc - tic)}")
        logger.info("---")

        # Reset counters
        epoch_start = time.time()
        start_time = epoch_start
        last_idx = 0

    # Save model hyperparameters with metrics
    if training_utils.is_local_primary(args) is True:
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
    if training_utils.is_local_primary(args) is True:
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
            "torchrun --nproc_per_node=2 -m birder.scripts.train_detection \\\n"
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
            "    --compile\n"
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
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("-t", "--tag", type=str, help="add model tag")
    parser.add_argument("--backbone", type=str, help="the neural network to used as backbone")
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
        help="start with pretrained version of specified backbone (will download if not found locally)",
    )
    parser.add_argument("--reset-head", default=False, action="store_true", help="reset the classification head")
    parser.add_argument(
        "--freeze-body",
        default=False,
        action="store_true",
        help="freeze all layers of the model except the classification head",
    )
    parser.add_argument("--freeze-backbone", default=False, action="store_true", help="freeze backbone")
    parser.add_argument("--freeze-backbone-stages", type=int, help="number of backbone stages to freeze")
    parser.add_argument(
        "--binary-mode",
        default=False,
        action="store_true",
        help="treat all objects as a single class (binary detection: object vs background)",
    )
    training_cli.add_optimization_args(parser, default_batch_size=16)
    training_cli.add_lr_wd_args(parser, backbone_lr=True)
    training_cli.add_lr_scheduler_args(parser)
    training_cli.add_training_schedule_args(parser)
    training_cli.add_detection_input_args(parser)
    training_cli.add_detection_data_aug_args(parser)
    training_cli.add_ema_args(parser, default_ema_steps=1, default_ema_decay=0.9998)
    training_cli.add_dataloader_args(
        parser,
        no_img_loader=True,
        default_num_workers=min(8, max(os.cpu_count() // 8, 2)),  # type: ignore[operator]
        ra_sampler=True,
    )
    training_cli.add_batch_norm_args(parser, backbone_freeze=True)
    training_cli.add_precision_args(parser)
    training_cli.add_compile_args(parser, backbone=True)
    training_cli.add_checkpoint_args(parser, pretrained=True)
    training_cli.add_distributed_args(parser)
    training_cli.add_logging_and_debug_args(parser, default_log_interval=20, fake_data=False)
    training_cli.add_detection_training_data_args(parser)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = str(args.data_path)
    args.val_path = str(args.val_path)
    args.size = cli.parse_size(args.size)

    # This will capture the common argument mistakes
    training_cli.common_args_validation(args)

    # Script specific checks
    if args.backbone is None:
        raise cli.ValidationError("--backbone is required")
    if registry.exists(args.network, task=Task.OBJECT_DETECTION) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")
    if registry.exists(args.backbone, net_type=DetectorBackbone) is False:
        raise cli.ValidationError(
            f"--backbone {args.network} not supported, see list-models tool for available options"
        )

    if args.freeze_backbone is True and args.freeze_backbone_stages is not None:
        raise cli.ValidationError("--freeze-backbone cannot be used with --freeze-backbone-stages")
    if args.freeze_backbone is True and args.freeze_body is True:
        raise cli.ValidationError("--freeze-backbone cannot be used with --freeze-body")
    if args.freeze_body is True and args.freeze_backbone_stages is not None:
        raise cli.ValidationError("--freeze-body cannot be used with --freeze-backbone-stages")
    if args.multiscale is True and args.aug_type != "birder":
        raise cli.ValidationError(f"--multiscale only supported with --aug-type birder, got {args.aug_type}")
    if args.backbone_pretrained is True and args.backbone_epoch is not None:
        raise cli.ValidationError("--backbone-pretrained cannot be used with --backbone-epoch")

    if args.model_dtype != "float32":  # NOTE: only float32 supported at this time
        raise cli.ValidationError(f"Only float32 supported for --model-dtype at this time, got {args.model_dtype}")


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    parser.set_defaults(**kwargs)
    args = parser.parse_args([])
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
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
