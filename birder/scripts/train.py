import argparse
import json
import logging
import math
import os
import sys
import time
import typing
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.amp
import torch.nn.functional as F
import torch.utils.data
import torchinfo
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.io import decode_image
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import training_utils
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.dataloader.webdataset import make_wds_loader
from birder.datasets.webdataset import make_wds_dataset
from birder.datasets.webdataset import prepare_wds_args
from birder.datasets.webdataset import wds_args_from_info
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import get_signature
from birder.transforms.classification import RGBMode
from birder.transforms.classification import get_mixup_cutmix
from birder.transforms.classification import get_rgb_stats
from birder.transforms.classification import inference_preset
from birder.transforms.classification import training_preset

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    #
    # Initialize
    #
    training_utils.init_distributed_mode(args)
    if args.size is None:
        args.size = registry.get_default_size(args.network)

    if args.aa is True:
        args.aug_level = -1

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
    training_transform = training_preset(args.size, args.aug_level, rgb_stats, args.resize_min_scale)
    val_transform = inference_preset(args.size, rgb_stats, 1.0)
    if args.wds is True:
        training_wds_path: str | list[str]
        val_wds_path: str | list[str]
        if args.wds_info_file is not None:
            (training_wds_path, training_size) = wds_args_from_info(args.wds_info_file, args.wds_training_split)
            (val_wds_path, val_size) = wds_args_from_info(args.wds_info_file, args.wds_val_split)
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
            cache_dir=args.wds_cache_dir,
        )
        validation_dataset = make_wds_dataset(
            val_wds_path,
            dataset_size=val_size,
            shuffle=False,
            samples_names=False,
            transform=val_transform,
            cache_dir=args.wds_cache_dir,
        )

        class_to_idx = fs_ops.read_class_file(args.wds_class_file)

    else:
        training_dataset = ImageFolder(args.data_path, transform=training_transform, loader=decode_image)
        validation_dataset = ImageFolder(args.val_path, transform=val_transform, loader=decode_image, allow_empty=True)
        assert training_dataset.class_to_idx == validation_dataset.class_to_idx
        class_to_idx = training_dataset.class_to_idx

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training on {len(training_dataset):,} samples")
    logger.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)
    batch_size: int = args.batch_size

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
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    sample_shape = (batch_size, args.channels, *args.size)  # B, C, H, W
    network_name = get_network_name(args.network, net_param=args.net_param, tag=args.tag)

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, class_to_idx_saved, training_states) = fs_ops.load_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            config=args.model_config,
            tag=args.tag,
            epoch=args.resume_epoch,
            new_size=args.size,
        )
        if args.reset_head is True:
            net.reset_classifier(len(class_to_idx))
        else:
            assert class_to_idx == class_to_idx_saved

    elif args.pretrained is True:
        (net, class_to_idx_saved, training_states) = fs_ops.load_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            config=args.model_config,
            tag=args.tag,
            epoch=None,
            new_size=args.size,
        )
        net.reset_classifier(len(class_to_idx))

    else:
        net = registry.net_factory(
            args.network,
            sample_shape[1],
            num_outputs,
            net_param=args.net_param,
            config=args.model_config,
            size=args.size,
        )
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
    if args.freeze_body is True:
        net.freeze(freeze_classifier=False, unfreeze_features=args.unfreeze_features)

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
    )

    # Loss criteria
    if args.bce_loss is True:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing_alpha)

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
        for _ in range(begin_epoch, epochs):
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

    #
    # Misc
    #

    # Define metrics
    # top_k = settings.TOP_K
    training_metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy("multiclass", num_classes=num_outputs),
            # f"top_{top_k}": torchmetrics.Accuracy("multiclass", num_classes=num_outputs, top_k=top_k),
            # "precision": torchmetrics.Precision("multiclass", num_classes=num_outputs, average="macro"),
            # "f1_score": torchmetrics.F1Score("multiclass", num_classes=num_outputs, average="macro"),
        },
        prefix="training_",
    ).to(device)
    validation_metrics = training_metrics.clone(prefix="validation_")

    # Print network summary
    net_for_info = net_without_ddp
    if args.compile is True and hasattr(net_without_ddp, "_orig_mod") is True:
        net_for_info = net_without_ddp._orig_mod  # pylint: disable=protected-access

    if args.no_summary is False:
        torchinfo.summary(
            net_for_info,
            device=device,
            input_size=sample_shape,
            dtypes=[model_dtype],
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
            verbose=1 if args.rank == 0 else 0,
        )

    # Training logs
    training_log_name = training_utils.training_log_name(network_name, device)
    training_log_path = settings.TRAINING_RUNS_PATH.joinpath(training_log_name)
    logger.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_signature(input_shape=sample_shape, num_outputs=num_outputs)
    if args.rank == 0:
        with torch.no_grad():
            summary_writer.add_graph(net_for_info, torch.rand(sample_shape, device=device, dtype=model_dtype))

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
        running_loss = training_utils.SmoothedValue(window_size=64)
        running_val_loss = training_utils.SmoothedValue()
        training_metrics.reset()
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
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)

            training_metrics(outputs, targets)

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                running_loss.synchronize_between_processes(device)
                training_metrics_dict = training_metrics.compute()
                if args.rank == 0:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )

                    for metric, value in training_metrics_dict.items():
                        summary_writer.add_scalars(
                            "performance",
                            {metric: value},
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

        for metric, value in training_metrics.compute().items():
            logger.info(f"Epoch {epoch}/{epochs-1} {metric}: {value:.4f}")

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
                validation_metrics(outputs, targets)

                # Update progress bar
                if args.rank == 0:
                    progress.update(n=batch_size * args.world_size)

        if args.rank == 0:
            progress.close()

        running_val_loss.synchronize_between_processes(device)
        epoch_val_loss = running_val_loss.global_avg
        validation_metrics_dict = validation_metrics.compute()

        # Learning rate scheduler update
        scheduler.step()
        if last_lr != max(scheduler.get_last_lr()):
            last_lr = max(scheduler.get_last_lr())
            logger.info(f"Updated learning rate to: {last_lr}")

        if args.rank == 0:
            summary_writer.add_scalars("loss", {"validation": epoch_val_loss}, epoch * len(training_dataset))
            for metric, value in validation_metrics_dict.items():
                summary_writer.add_scalars("performance", {metric: value}, epoch * len(training_dataset))

            # Epoch validation metrics
            logger.info(f"Epoch {epoch}/{epochs-1} validation_loss: {epoch_val_loss:.4f}")
            for metric, value in validation_metrics_dict.items():
                logger.info(f"Epoch {epoch}/{epochs-1} {metric}: {value:.4f}")

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

        # Epoch timing
        toc = time.time()
        (minutes, seconds) = divmod(toc - tic, 60)
        logger.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logger.info("---")

    # Save model hyperparameters with metrics
    if args.rank == 0:
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
        metrics = training_metrics.compute()
        val_metrics = validation_metrics.compute()
        summary_writer.add_hparams(
            {**vars(args), "training_samples": len(training_dataset)},
            {
                "hparam/acc": metrics["training_accuracy"],
                "hparam/val_acc": val_metrics["validation_accuracy"],
            },
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
        description="Train classification model",
        epilog=(
            "Usage examples\n"
            "==============\n"
            "Training on image directory in the default location can be as simple as:\n"
            "python train.py --network densenet_161 --lr-scheduler cosine --batch-size 64 --smoothing-alpha 0.1\n"
            "\n"
            "A more complicated ImageNet example on 2 GPU's, using a remote webdataset:\n"
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
            "    --aug-level 4 \\\n"
            "    --ra-sampler --ra-reps 2 \\\n"
            "    --rgb-mode imagenet \\\n"
            "    --bce-loss --bce-threshold 0.2 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
            "    --wds \\\n"
            "    --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt \\\n"
            "    --wds-info-file https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/_info.json \\\n"
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
            "    --aug-level 4 \\\n"
            "    --amp --compile\n"
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
    parser.add_argument(
        "--pretrained",
        default=False,
        action="store_true",
        help="start with pretrained version of specified network, reset the classification head",
    )
    parser.add_argument("--reset-head", default=False, action="store_true", help="reset the classification head")
    parser.add_argument(
        "--freeze-body",
        default=False,
        action="store_true",
        help="freeze all layers of the model except the classification head",
    )
    parser.add_argument(
        "--unfreeze-features",
        default=False,
        action="store_true",
        help="unfreeze features layer (only relevant when freezing body)",
    )
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-opt", default=False, action="store_true", help="enable compilation for optimizer step"
    )
    training_utils.add_optimizer_args(parser)
    parser.add_argument("--lr", type=float, default=0.1, help="base learning rate")
    parser.add_argument(
        "--lr-scale", type=int, help="reference batch size for LR scaling, if provided, LR will be scaled accordingly"
    )
    parser.add_argument(
        "--lr-scale-type", type=str, choices=["linear", "sqrt"], default="linear", help="learning rate scaling type"
    )
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
        "--freeze-bn",
        default=False,
        action="store_true",
        help="freeze all batch statistics and affine parameters of batchnorm2d layers",
    )
    parser.add_argument("--sync-bn", default=False, action="store_true", help="use synchronized BatchNorm")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N", help="number of warmup epochs")
    parser.add_argument("--smoothing-alpha", type=float, default=0.0, help="label smoothing alpha")
    parser.add_argument("--mixup-alpha", type=float, help="mixup alpha")
    parser.add_argument("--cutmix", default=False, action="store_true", help="enable cutmix")
    parser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=2,
        help="magnitude of augmentations (0 off -> 4 highest)",
    )
    parser.add_argument("--aa", default=False, action="store_true", help="use AutoAugment policy (ignoring aug-level)")
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--resize-min-scale", type=float, help="random resize min scale")
    parser.add_argument("--bce-loss", default=False, action="store_true", help="enable BCE loss")
    parser.add_argument("--bce-threshold", type=float, default=0.0, help="threshold for binarizing soft BCE targets")
    parser.add_argument("--epochs", type=int, default=100, metavar="N", help="number of training epochs")
    parser.add_argument(
        "--stop-epoch", type=int, metavar="N", help="epoch to stop the training at (multi step training)"
    )
    parser.add_argument("--save-frequency", type=int, default=5, metavar="N", help="frequency of model saving")
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
    parser.add_argument(
        "--ra-reps", type=int, default=3, metavar="N", help="number of repetitions for Repeated Augmentation"
    )
    parser.add_argument("-t", "--tag", type=str, help="add training logs tag")
    parser.add_argument(
        "--log-interval", type=int, default=50, metavar="N", help="how many steps between summary writes"
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=max(os.cpu_count() // 4, 4),  # type: ignore[operator]
        metavar="N",
        help="number of preprocessing workers",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    parser.add_argument("--drop-last", default=False, action="store_true", help="drop the last incomplete batch")
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    parser.add_argument("--amp", default=False, action="store_true", help="use torch.amp for mixed precision training")
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
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
    parser.add_argument("--no-summary", default=False, action="store_true", help="don't print model summary")
    parser.add_argument(
        "--val-path", type=str, default=str(settings.VALIDATION_DATA_PATH), help="validation directory path"
    )
    parser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="training directory path"
    )
    training_utils.add_wds_args(parser)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = str(args.data_path)
    args.val_path = str(args.val_path)
    assert args.network is not None
    assert (
        args.pretrained is False or args.resume_epoch is None
    ), "Cannot set resume epoch while starting from a pretrained network"
    assert (
        args.stop_epoch is None or args.stop_epoch < args.epochs
    ), "Stop epoch must be smaller than total number of epochs"
    assert 0.5 > args.smoothing_alpha >= 0, "Smoothing alpha must be in range of [0, 0.5)"
    assert (
        args.load_states is False or args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert (
        args.load_scheduler is False or args.resume_epoch is not None
    ), "Load scheduler must be from resumed training (--resume-epoch)"
    assert args.wds is False or args.ra_sampler is False, "Repeated Augmentation not currently supported with wds"
    assert args.wds is False or args.wds_class_file is not None, "Must set a class file"
    assert (
        registry.exists(args.network, task=Task.IMAGE_CLASSIFICATION) is True
    ), "Unknown network, see list-models tool for available options"
    assert args.freeze_bn is False or args.sync_bn is False, "Cannot freeze-bn and sync-bn are mutually exclusive"
    assert args.amp is False or args.model_dtype == "float32"
    assert args.resize_min_scale is None or args.resize_min_scale < 1.0
    assert args.bce_loss is False or args.smoothing_alpha == 0.0
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
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.wds_cache_dir is not None and Path(args.wds_cache_dir).exists() is False:
        logger.info(f"Creating {args.wds_cache_dir} directory...")
        Path(args.wds_cache_dir).mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    logger = logging.getLogger(__spec__.name)
    main()
