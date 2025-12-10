import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FakeData
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader  # Slower but Handles external dataset quirks better
from tqdm import tqdm

import birder
from birder.common import cli
from birder.common import fs_ops
from birder.common import training_cli
from birder.common import training_utils
from birder.common.lib import format_duration
from birder.common.lib import get_network_name
from birder.common.lib import set_random_seeds
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import HierarchicalImageFolder
from birder.data.datasets.directory import tv_loader
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import get_mixup_cutmix
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import get_signature

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    #
    # Initialize
    #
    training_utils.init_distributed_mode(args)
    logger.info(f"Starting training, birder version: {birder.__version__}, pytorch version: {torch.__version__}")
    training_utils.log_git_info()

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

    if args.seed is not None:
        set_random_seeds(args.seed)

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

    training_transform = training_utils.get_training_transform(args)
    val_transform = inference_preset(args.size, rgb_stats, 1.0)
    if args.use_fake_data is True:
        logger.warning("Using fake data")
        training_dataset = FakeData(10000, (args.channels, *args.size), num_classes=10, transform=training_transform)
        validation_dataset = FakeData(1000, (args.channels, *args.size), num_classes=10, transform=val_transform)
        class_to_idx = {str(i): i for i in range(10)}

    elif args.wds is True:
        training_wds_path: str | list[str]
        val_wds_path: str | list[str]
        if args.wds_info is not None:
            (training_wds_path, training_size) = wds_args_from_info(args.wds_info, args.wds_training_split)
            (val_wds_path, val_size) = wds_args_from_info(args.wds_info, args.wds_val_split)
            if args.wds_train_size is not None:
                training_size = args.wds_train_size
            if args.wds_val_size is not None:
                val_size = args.wds_val_size
        else:
            (training_wds_path, training_size) = prepare_wds_args(args.data_path, args.wds_train_size, device)
            (val_wds_path, val_size) = prepare_wds_args(args.val_path, args.wds_val_size, device)

        training_dataset = make_wds_dataset(
            training_wds_path,
            dataset_size=training_size,
            shuffle=True,
            samples_names=False,
            transform=training_transform,
            img_loader=args.img_loader,
            cache_dir=args.wds_cache_dir,
        )
        validation_dataset = make_wds_dataset(
            val_wds_path,
            dataset_size=val_size,
            shuffle=False,
            samples_names=False,
            transform=val_transform,
            img_loader=args.img_loader,
            cache_dir=args.wds_cache_dir,
        )

        class_to_idx = fs_ops.read_class_file(args.wds_class_file)

    else:
        if args.hierarchical is True:
            dataset_cls = HierarchicalImageFolder
        else:
            dataset_cls = ImageFolder

        training_dataset = dataset_cls(
            args.data_path, transform=training_transform, loader=pil_loader if args.img_loader == "pil" else tv_loader
        )
        validation_dataset = dataset_cls(
            args.val_path,
            transform=val_transform,
            loader=pil_loader if args.img_loader == "pil" else tv_loader,
            allow_empty=True,
        )
        assert training_dataset.class_to_idx == validation_dataset.class_to_idx
        class_to_idx = training_dataset.class_to_idx

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training on {len(training_dataset):,} samples")
    logger.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)
    batch_size: int = args.batch_size
    model_ema_steps: int = args.model_ema_steps * args.grad_accum_steps
    logger.debug(f"Effective batch size = {args.batch_size * args.grad_accum_steps * args.world_size}")

    # Set data iterators
    if args.mixup_alpha is not None or args.cutmix is True:
        logger.debug("Mixup / cutmix collate activated")
        t = get_mixup_cutmix(args.mixup_alpha, num_outputs, args.cutmix)

        def collate_fn(batch: Any) -> Any:
            return t(*default_collate(batch))

    else:
        collate_fn = None  # type: ignore

    # Data loaders and samplers
    (train_sampler, validation_sampler) = training_utils.get_samplers(args, training_dataset, validation_dataset)

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=collate_fn,
            world_size=args.world_size,
            pin_memory=True,
            drop_last=args.drop_last,
            shuffle=args.wds_extra_shuffle,
        )

        validation_loader = make_wds_loader(
            validation_dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
            world_size=args.world_size,
            pin_memory=True,
        )

    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=args.drop_last,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=True,
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
    network_name = get_network_name(args.network, tag=args.tag)

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, class_to_idx_saved, training_states) = fs_ops.load_checkpoint(
            device,
            args.network,
            config=args.model_config,
            tag=args.tag,
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
        (net, class_to_idx_saved, training_states) = fs_ops.load_checkpoint(
            device,
            args.network,
            config=args.model_config,
            tag=args.tag,
            epoch=None,
            new_size=args.size,
            strict=not args.non_strict_weights,
        )
        if args.reset_head is True:
            net.reset_classifier(len(class_to_idx))
        else:
            assert class_to_idx == class_to_idx_saved

    else:
        net = registry.net_factory(args.network, sample_shape[1], num_outputs, config=args.model_config, size=args.size)
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
    if args.freeze_body is True:
        net.freeze(freeze_classifier=False, unfreeze_features=args.unfreeze_features)
    elif args.freeze_stages is not None:
        net.freeze_stages(up_to_stage=args.freeze_stages)

    if args.freeze_bn is True:
        net = training_utils.freeze_batchnorm2d(net)
    elif args.sync_bn is True and args.distributed is True:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile network
    if args.compile is True:
        net = torch.compile(net)

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
    )

    # Loss criteria
    if args.bce_loss is True:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing_alpha)

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

    signature = get_signature(input_shape=sample_shape, num_outputs=num_outputs)
    file_handler: logging.Handler = logging.NullHandler()
    if training_utils.is_local_primary(args) is True:
        with torch.no_grad():
            summary_writer.add_graph(net_for_info, torch.rand(sample_shape, device=device, dtype=model_dtype))

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
        running_loss = training_utils.SmoothedValue(window_size=64)
        running_val_loss = training_utils.SmoothedValue()
        train_accuracy = training_utils.SmoothedValue(window_size=64)
        val_accuracy = training_utils.SmoothedValue()

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
        for i, (inputs, targets) in enumerate(training_loader):
            inputs = inputs.to(device, dtype=model_dtype, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss_targets = targets
            if args.bce_loss is True:
                if loss_targets.ndim == 1:
                    loss_targets = F.one_hot(loss_targets, num_classes=num_outputs)  # pylint: disable=not-callable

                loss_targets = loss_targets.gt(args.bce_threshold).to(dtype=inputs.dtype)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                outputs = net(inputs)
                loss = criterion(outputs, loss_targets)

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
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)

            train_accuracy.update(training_utils.accuracy(targets, outputs.detach()))

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
                train_accuracy.synchronize_between_processes(device)
                with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                    log.info(
                        f"[Trn] Epoch {epoch}/{epochs-1}, step {i+1}/{last_batch_idx+1}  "
                        f"Loss: {running_loss.avg:.4f}  "
                        f"Elapsed: {format_duration(time_now-epoch_start)}  "
                        f"ETA: {format_duration(estimated_time_to_finish_epoch)}  "
                        f"T: {time_cost:.1f}s  "
                        f"R: {rate:.1f} samples/s  "
                        f"LR: {cur_lr:.4e}"
                    )
                    log.info(
                        f"[Trn] Epoch {epoch}/{epochs-1}, step {i+1}/{last_batch_idx+1}  "
                        f"Accuracy: {train_accuracy.avg:.4f}"
                    )

                if training_utils.is_local_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )
                    summary_writer.add_scalars(
                        "performance",
                        {"training_accuracy": train_accuracy.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )

            # Update progress bar
            progress.update(n=batch_size * args.world_size)

        progress.close()

        # Epoch training metrics
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} training_loss: {running_loss.global_avg:.4f}")
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} training_accuracy: {train_accuracy.global_avg:.4f}")

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
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, dtype=model_dtype, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                loss_targets = targets
                if args.bce_loss is True:
                    loss_targets = F.one_hot(loss_targets, num_classes=num_outputs)  # pylint: disable=not-callable
                    loss_targets = loss_targets.to(dtype=inputs.dtype)

                with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                    outputs = eval_model(inputs)
                    val_loss = criterion(outputs, loss_targets)

                # Statistics
                running_val_loss.update(val_loss.detach())
                val_accuracy.update(training_utils.accuracy(targets, outputs), n=outputs.size(0))

                # Update progress bar
                progress.update(n=batch_size * args.world_size)

        time_now = time.time()
        rate = len(validation_dataset) / (time_now - epoch_start)
        with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
            log.info(
                f"[Val] Epoch {epoch}/{epochs-1} "
                f"Elapsed: {format_duration(time_now-epoch_start)}  "
                f"R: {rate:.1f} samples/s"
            )

        progress.close()

        running_val_loss.synchronize_between_processes(device)
        val_accuracy.synchronize_between_processes(device)
        epoch_val_loss = running_val_loss.global_avg
        epoch_val_accuracy = val_accuracy.global_avg

        # Write statistics
        if training_utils.is_local_primary(args) is True:
            summary_writer.add_scalars("loss", {"validation": epoch_val_loss}, epoch * len(training_dataset))
            summary_writer.add_scalars(
                "performance", {"validation_accuracy": epoch_val_accuracy}, epoch * len(training_dataset)
            )

        # Epoch validation metrics
        logger.info(f"[Val] Epoch {epoch}/{epochs-1} validation_loss: {epoch_val_loss:.4f}")
        logger.info(f"[Val] Epoch {epoch}/{epochs-1} validation_accuracy: {epoch_val_accuracy:.4f}")

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
        logger.info(f"Total time: {format_duration(toc - tic)}")
        logger.info("---")

        # Reset counters
        epoch_start = time.time()
        start_time = epoch_start
        last_idx = 0

    # Save model hyperparameters with metrics
    if training_utils.is_local_primary(args) is True:
        # Replace list/dict based args
        if args.opt_betas is not None:
            for idx, beta in enumerate(args.opt_betas):
                setattr(args, f"opt_betas_{idx}", beta)

            del args.opt_betas

        if args.lr_steps is not None:
            args.lr_steps = json.dumps(args.lr_steps)
        if args.model_config is not None:
            args.model_config = json.dumps(args.model_config)
        if args.size is not None:
            args.size = json.dumps(args.size)

        # Save all args
        summary_writer.add_hparams(
            {**vars(args), "training_samples": len(training_dataset)},
            {
                "hparam/acc": train_accuracy.global_avg,
                "hparam/val_acc": val_accuracy.global_avg,
            },
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
        description="Train classification model",
        epilog=(
            "Usage examples\n"
            "==============\n"
            "Training on image directory in the default location can be as simple as:\n"
            "python train.py --network densenet_161 --lr-scheduler cosine --batch-size 64 --smoothing-alpha 0.1\n"
            "\n"
            "A more complicated ImageNet example on 2 GPUs, using a remote webdataset:\n"
            "torchrun --nproc_per_node=2 train.py \\\n"
            "    --network resnet_v2_50 \\\n"
            "    --tag imagenet1k \\\n"
            "    --opt lamb \\\n"
            "    --lr 0.005 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --lr-cosine-min 1e-7 \\\n"
            "    --warmup-epochs 5 \\\n"
            "    --epochs 300 \\\n"
            "    --wd 0.02 \\\n"
            "    --grad-accum-steps 4 \\\n"
            "    --mixup-alpha 0.1 \\\n"
            "    --cutmix \\\n"
            "    --aug-type ra \\\n"
            "    --re-prob 0.25 \\\n"
            "    --rgb-mode imagenet \\\n"
            "    --bce-loss --bce-threshold 0.2 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
            "    --wds \\\n"
            "    --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt \\\n"
            "    --wds-info https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/_info.json \\\n"
            "    --wds-training-split train\n"
            "\n"
            "The script is also available as module:\n"
            "torchrun --nproc_per_node=2 -m birder.scripts.train \\\n"
            "    --network regnet_y_8g \\\n"
            "    --lr 0.4 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --warmup-epochs 5 \\\n"
            "    --wd 0.00005 \\\n"
            "    --smoothing-alpha 0.1 \\\n"
            "    --mixup-alpha 0.2 \\\n"
            "    --cutmix \\\n"
            "    --aug-level 8 \\\n"
            "    --ra-sampler --ra-reps 2 \\\n"
            "    --amp --compile\n"
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
    parser.add_argument("--reset-head", default=False, action="store_true", help="reset the classification head")
    parser.add_argument(
        "--freeze-body",
        default=False,
        action="store_true",
        help="freeze all layers of the model except the classification head",
    )
    parser.add_argument("--freeze-stages", type=int, help="number of model stages to freeze (for supported models)")
    parser.add_argument(
        "--unfreeze-features",
        default=False,
        action="store_true",
        help="unfreeze features layer (only relevant when freezing body)",
    )

    group = parser.add_argument_group("Loss parameters")
    group.add_argument("--smoothing-alpha", type=float, default=0.0, help="label smoothing alpha")
    group.add_argument("--bce-loss", default=False, action="store_true", help="enable BCE loss")
    group.add_argument("--bce-threshold", type=float, default=0.0, help="threshold for binarizing soft BCE targets")
    training_cli.add_optimization_args(parser)
    training_cli.add_lr_wd_args(parser)
    training_cli.add_lr_scheduler_args(parser)
    training_cli.add_training_schedule_args(parser)
    training_cli.add_input_args(parser)
    training_cli.add_data_aug_args(parser, mixup_cutmix=True)
    training_cli.add_ema_args(parser)
    training_cli.add_dataloader_args(parser, ra_sampler=True)
    training_cli.add_batch_norm_args(parser)
    training_cli.add_precision_args(parser)
    training_cli.add_compile_args(parser)
    training_cli.add_checkpoint_args(parser, default_save_frequency=5, pretrained=True)
    training_cli.add_distributed_args(parser)
    training_cli.add_logging_and_debug_args(parser)
    training_cli.add_training_data_args(parser)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = str(args.data_path)
    args.val_path = str(args.val_path)
    args.size = cli.parse_size(args.size)

    # This will capture the common argument mistakes
    training_cli.common_args_validation(args)

    # Script specific checks
    if registry.exists(args.network, task=Task.IMAGE_CLASSIFICATION) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")

    if args.smoothing_alpha < 0 or args.smoothing_alpha >= 0.5:
        raise cli.ValidationError(f"--smoothing-alpha must be in range of [0, 0.5), got {args.smoothing_alpha}")
    if args.bce_loss is True and args.smoothing_alpha != 0.0:
        raise cli.ValidationError("--bce-loss can only be used with --smoothing-alpha 0.0")

    if args.freeze_body is True and args.freeze_stages is not None:
        raise cli.ValidationError("--freeze-body cannot be used with --freeze-stages")
    if args.freeze_stages is not None and registry.exists(args.network, net_type=DetectorBackbone) is False:
        raise cli.ValidationError(
            "--freeze-stages only supported on detector backbone type networks, "
            "see list-models tool for available options"
        )


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
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.wds_cache_dir is not None and Path(args.wds_cache_dir).exists() is False:
        logger.info(f"Creating {args.wds_cache_dir} directory...")
        Path(args.wds_cache_dir).mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
