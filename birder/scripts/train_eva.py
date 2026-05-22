"""
EVA-style masked image modeling, adapted from
https://github.com/baaivision/EVA/blob/master/EVA-01/eva/engine_for_pretraining.py

Paper "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale",
https://arxiv.org/abs/2211.07636
"""

# Reference license: MIT

import argparse
import logging
import math
import sys
import time
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import training_cli
from birder.common import training_utils
from birder.common.lib import format_duration
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.common.lib import get_size_from_signature
from birder.common.masking import BlockMasking
from birder.common.masking import Masking
from birder.common.masking import UniformMasking
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.fake import FakeDataWithPaths
from birder.data.datasets.webdataset import WDSImageDecoder
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import get_rgb_stats
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import get_signature
from birder.net.mim.base import get_mim_signature
from birder.net.mim.eva import EVA

logger = logging.getLogger(__name__)

ImageLoader = Callable[[str], Any]
ImageTransform = Callable[[Any], torch.Tensor]
TransformFactory = Callable[[argparse.Namespace], ImageTransform]


@dataclass(frozen=True)
class TrainOverrides:
    training_transform: Optional[TransformFactory] = None
    image_loader: Optional[ImageLoader] = None
    wds_image_decoder: Optional[WDSImageDecoder] = None


class TrainCollator:
    def __init__(self, mask_generator: Callable[[int], torch.Tensor]) -> None:
        self.collator = torch.utils.data.default_collate
        self.mask_generator = mask_generator

    def __call__(self, batch: Any) -> tuple[Any, torch.Tensor]:
        B = len(batch)
        collated_batch = self.collator(batch)
        masks = self.mask_generator(B)

        return (collated_batch, masks)


def get_mask_generator(masking: str, mask_size: tuple[int, int], mask_ratio: float, min_mask_size: int) -> Masking:
    if masking == "block":
        num_patches = mask_size[0] * mask_size[1]
        max_num_patches = int(num_patches * mask_ratio)
        min_masking_patches = int(num_patches * max(mask_ratio - 0.1, 0.0))
        min_num_patches = min(16, max_num_patches)
        return BlockMasking(
            mask_size,
            min_num_patches=min_num_patches,
            max_num_patches=max_num_patches,
            min_aspect=0.33,
            max_aspect=3.33,
            min_masking_patches=min_masking_patches,
        )

    if masking == "uniform":
        return UniformMasking(mask_size, mask_ratio, min_mask_size=min_mask_size)

    raise ValueError(f"Unsupported masking strategy: {masking}")


def teacher_tokens(teacher: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    features = teacher.forward_features(x)
    if features.ndim == 3:
        num_special_tokens = getattr(teacher, "num_special_tokens", 0)
        return features[:, num_special_tokens:]

    if features.ndim == 4:
        return features.flatten(2).permute(0, 2, 1)

    raise RuntimeError(f"Unsupported teacher feature tensor: {features.size()}")


def train(args: argparse.Namespace, overrides: Optional[TrainOverrides] = None) -> None:
    if overrides is None:
        overrides = TrainOverrides()

    #
    # Initialize
    #
    device, device_id, disable_tqdm = training_utils.init_training(args, logger)

    # Using the teacher rgb values for the student
    teacher, (_, signature, rgb_stats, *_) = fs_ops.load_model(
        device,
        args.teacher,
        config=args.teacher_model_config,
        tag=args.teacher_tag,
        epoch=args.teacher_epoch,
        new_size=args.size,
        inference=True,
    )
    if args.size is None:
        args.size = get_size_from_signature(signature)

    logger.info(f"Using size={args.size}")

    batch_size: int = args.batch_size
    grad_accum_steps: int = args.grad_accum_steps
    logger.debug(f"Effective batch size = {batch_size * grad_accum_steps * args.world_size}")

    #
    # Initialize network
    #
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    sample_shape = (batch_size, args.channels, *args.size)  # B, C, H, W
    backbone_name = get_network_name(args.network, tag="eva")
    if args.tag is not None:
        backbone_name = f"{backbone_name}-{args.tag}"

    network_name = get_mim_network_name("eva", encoder=args.network, tag=args.tag)

    backbone = registry.net_factory(args.network, 0, sample_shape[1], config=args.model_config, size=args.size)
    teacher.to(dtype=model_dtype)

    if hasattr(teacher, "max_stride") is True and backbone.max_stride != teacher.max_stride:
        raise RuntimeError(
            f"EVA requires student and teacher token grids to match, got student stride {backbone.max_stride} "
            f"and teacher stride {teacher.max_stride}"
        )

    with torch.no_grad():
        # Infer from forward_features as encoding_size is not available on all networks.
        target_tokens = teacher_tokens(teacher, torch.zeros(sample_shape, device=device, dtype=model_dtype))

    net = EVA(
        backbone,
        config={"teacher_dim": target_tokens.size(-1)},
        size=args.size,
        mask_ratio=args.mask_ratio,
        min_mask_size=args.min_mask_size,
    )

    #
    # Data
    #
    logger.debug(f"Using RGB stats: {rgb_stats}")
    training_rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    if training_rgb_stats != rgb_stats:
        logger.warning(f"Training RGB stats {training_rgb_stats}, but teacher was saved with {rgb_stats}")

    mask_size = (args.size[0] // net.encoder.max_stride, args.size[1] // net.encoder.max_stride)
    mask_generator = get_mask_generator(args.masking, mask_size, net.mask_ratio, args.min_mask_size)
    mask_collator = TrainCollator(mask_generator)
    if overrides.training_transform is not None:
        training_transform = overrides.training_transform(args)
    else:
        training_transform = training_utils.get_training_transform(args)

    if args.use_fake_data is True:
        logger.warning("Using fake data")
        training_dataset = FakeDataWithPaths(
            10000, (args.channels, *args.size), num_classes=10, transform=training_transform
        )

    elif args.wds is True:
        if overrides.wds_image_decoder is not None:
            wds_image_decoder = overrides.wds_image_decoder
        else:
            wds_image_decoder = args.img_loader

        wds_path: str | list[str]
        if args.wds_info is not None:
            wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            wds_path, dataset_size = prepare_wds_args(args.data_path[0], args.wds_size, device)

        training_dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=True,
            transform=training_transform,
            image_decoder=wds_image_decoder,
            channels=args.channels,
            cls_key=None,
            cache_dir=args.wds_cache_dir,
        )

    else:
        if overrides.image_loader is not None:
            image_loader = overrides.image_loader
        else:
            image_loader = get_image_loader(args.img_loader, args.channels)

        training_dataset = make_image_dataset(
            args.data_path,
            {},
            transforms=training_transform,
            loader=image_loader,
        )

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training dataset has {len(training_dataset):,} samples")

    # Data loaders and samplers
    virtual_epoch_mode = args.steps_per_epoch is not None
    train_sampler, _ = training_utils.get_samplers(
        args, training_dataset, validation_dataset=None, infinite=virtual_epoch_mode
    )

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=mask_collator,
            world_size=args.world_size,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
            persistent_workers=args.persistent_workers,
            shuffle=False,
            infinite=virtual_epoch_mode,
        )

    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=mask_collator,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
            persistent_workers=args.persistent_workers,
        )

    if virtual_epoch_mode is True:
        optimizer_steps_per_epoch = args.steps_per_epoch
        epoch_num_batches = args.steps_per_epoch * grad_accum_steps
        epoch_samples = epoch_num_batches * batch_size * args.world_size
        logger.debug(f"Virtual epoch has {epoch_samples:,} samples")
    else:
        optimizer_steps_per_epoch = math.ceil(len(training_loader) / grad_accum_steps)
        epoch_num_batches = len(training_loader)
        epoch_samples = len(training_dataset)

    last_batch_idx = epoch_num_batches - 1
    last_accum_steps = epoch_num_batches % grad_accum_steps
    if last_accum_steps == 0:
        last_accum_steps = grad_accum_steps

    last_accum_start_idx = epoch_num_batches - last_accum_steps
    begin_epoch = 1
    epochs = args.epochs + 1
    args.stop_epoch = training_utils.normalize_stop_epoch(epochs, args.stop_epoch)

    logger.debug(
        f"Epoch has {epoch_num_batches} iterations ({optimizer_steps_per_epoch} steps), "
        f"virtual mode={virtual_epoch_mode}"
    )

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        net, training_states = fs_ops.load_simple_checkpoint(
            device, net, network_name, epoch=args.resume_epoch, strict=not args.non_strict_weights
        )

    else:
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
    if args.freeze_encoder is True:
        net.encoder.freeze(freeze_classifier=False)
    elif args.freeze_encoder_stages is not None:
        net.encoder.freeze_stages(up_to_stage=args.freeze_encoder_stages)
    elif args.freeze_encoder_layers is not None:
        frozen_layers = training_utils.freeze_layers_by_block_group_regex(net.encoder, args.freeze_encoder_layers)
        logger.info(f"Froze {frozen_layers} encoder layers using block_group_regex")

    if args.freeze_bn is True:
        net = training_utils.freeze_batchnorm2d(net)
    elif args.sync_bn is True and args.distributed is True:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile networks
    if args.compile is True:
        net = torch.compile(net, fullgraph=args.compile_fullgraph, mode=args.compile_mode)
        teacher.forward_features = torch.compile(
            teacher.forward_features, fullgraph=args.compile_fullgraph, mode=args.compile_mode
        )
    elif args.compile_teacher is True:
        teacher.forward_features = torch.compile(
            teacher.forward_features, fullgraph=args.compile_fullgraph, mode=args.compile_mode
        )

    #
    # Optimizer, learning rate scheduler and training parameter groups
    #

    # Learning rate scaling
    lr = training_utils.scale_lr(args)

    # Training parameter groups
    custom_keys_weight_decay = training_utils.get_wd_custom_keys(args)
    parameters = training_utils.optimizer_parameter_groups(
        net,
        args.wd,
        base_lr=lr,
        norm_weight_decay=args.norm_wd,
        custom_keys_weight_decay=custom_keys_weight_decay,
        custom_layer_weight_decay=args.custom_layer_wd,
        layer_decay=args.layer_decay,
        layer_decay_min_scale=args.layer_decay_min_scale,
        layer_decay_no_opt_scale=args.layer_decay_no_opt_scale,
        bias_lr=args.bias_lr,
        custom_layer_lr_scale=args.custom_layer_lr_scale,
    )

    if args.lr_scheduler_update == "epoch":
        step_update = False
        scheduler_steps_per_epoch = 1
    elif args.lr_scheduler_update == "step":
        step_update = True
        scheduler_steps_per_epoch = optimizer_steps_per_epoch
    else:
        raise ValueError("Unsupported lr_scheduler_update")

    # Optimizer and learning rate scheduler
    optimizer = training_utils.get_optimizer(parameters, lr, args)
    scheduler = training_utils.get_scheduler(optimizer, scheduler_steps_per_epoch, args)
    if args.compile_opt is True:
        optimizer.step = torch.compile(optimizer.step, fullgraph=False)

    # Gradient scaler and AMP related tasks
    scaler, amp_dtype = training_utils.get_amp_scaler(args.amp, args.amp_dtype)

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
            for _ in range(scheduler_steps_per_epoch):
                lrs.append(float(max(scheduler.get_last_lr())))
                scheduler.step()

        plt.plot(
            np.linspace(begin_epoch, epochs, scheduler_steps_per_epoch * (epochs - begin_epoch), endpoint=False), lrs
        )
        plt.show()
        raise SystemExit(0)

    #
    # Distributed (DDP)
    #
    net_without_ddp = net
    no_sync_cm = nullcontext
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_parameters,
            broadcast_buffers=not args.no_broadcast_buffers,
        )
        no_sync_cm = net.no_sync
        net_without_ddp = net.module

    model_to_save = net_without_ddp
    if args.compile is True and hasattr(model_to_save, "_orig_mod") is True:
        model_to_save = model_to_save._orig_mod

    #
    # Misc
    #

    # Print network summary
    net_for_info = net_without_ddp
    if args.compile is True and hasattr(net_without_ddp, "_orig_mod") is True:
        net_for_info = net_without_ddp._orig_mod

    if args.no_summary is False:
        summary = torchinfo.summary(
            net_for_info,
            device=device,
            input_data=(
                torch.rand(sample_shape),
                torch.rand((batch_size, target_tokens.size(1), target_tokens.size(2))),
                mask_generator(batch_size),
            ),
            dtypes=[model_dtype, model_dtype, model_dtype],
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

    signature = get_mim_signature(input_shape=sample_shape)
    backbone_signature = get_signature(input_shape=sample_shape, num_outputs=0)
    file_handler: logging.Handler = logging.NullHandler()
    if training_utils.is_global_primary(args) is True:
        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        file_handler = training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
        training_utils.write_training_args_json(training_log_path, args)
        training_utils.write_training_data_json(training_log_path, {"training_samples": len(training_dataset)})

    #
    # Training loop
    #
    if virtual_epoch_mode is True:
        train_iter = iter(training_loader)

    running_loss = training_utils.SmoothedValue(window_size=64)

    logger.info(f"Starting training with learning rate of {last_lr}")
    for epoch in range(begin_epoch, args.stop_epoch):
        tic = time.time()
        net.train()
        teacher.eval()

        # Clear metrics
        running_loss.clear()

        if args.distributed is True or virtual_epoch_mode is True:
            train_sampler.set_epoch(epoch)

        progress = tqdm(
            desc=f"Epoch {epoch}/{epochs-1}",
            total=epoch_samples,
            leave=False,
            disable=disable_tqdm,
            unit="samples",
            initial=0,
        )

        # Zero the parameter gradients
        optimizer.zero_grad()

        epoch_start = time.time()
        start_time = epoch_start
        last_idx = -1
        batch_iter: Iterator[tuple[int, Any]]
        if virtual_epoch_mode is True:
            batch_iter = ((i, next(train_iter)) for i in range(epoch_num_batches))
        else:
            batch_iter = enumerate(training_loader)

        for i, ((_, images, _), masks) in batch_iter:
            images = images.to(device, dtype=model_dtype, non_blocking=True)
            masks = masks.to(device, dtype=model_dtype, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)
            sync_context = no_sync_cm if optimizer_update is False else nullcontext
            if i >= last_accum_start_idx:
                effective_accum_steps = last_accum_steps
            else:
                effective_accum_steps = grad_accum_steps

            # Forward and backward
            with sync_context():
                with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                    with torch.no_grad():
                        target_tokens = teacher_tokens(teacher, images)

                    outputs: dict[str, torch.Tensor] = net(images, target_tokens, masks)
                    raw_loss = outputs["loss"]

                loss = raw_loss / effective_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if optimizer_update is True:
                if scaler is not None:
                    if args.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    optimizer.step()

                optimizer.zero_grad()
                if step_update is True:
                    scheduler.step()

            # Statistics
            running_loss.update(raw_loss.detach())

            # Write statistics
            if (i + 1) % args.log_interval == 0 or i == last_batch_idx:
                time_now = time.time()
                time_cost = time_now - start_time
                iters_processed_in_interval = i - last_idx
                rate = iters_processed_in_interval * (batch_size * args.world_size) / time_cost

                avg_time_per_iter = time_cost / iters_processed_in_interval
                remaining_iters_in_epoch = last_batch_idx - i
                estimated_time_to_finish_epoch = remaining_iters_in_epoch * avg_time_per_iter

                start_time = time_now
                last_idx = i
                cur_lr = float(max(scheduler.get_last_lr()))

                running_loss.synchronize_between_processes(device)
                with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                    log.info(
                        f"[Trn] Epoch {epoch}/{epochs-1}, iter {i+1}/{last_batch_idx+1}  "
                        f"Loss: {running_loss.avg:.4f}  "
                        f"Elapsed: {format_duration(time_now-epoch_start)}  "
                        f"ETA: {format_duration(estimated_time_to_finish_epoch)}  "
                        f"T: {time_cost:.1f}s  "
                        f"R: {rate:.1f} samples/s  "
                        f"LR: {cur_lr:.4e}"
                    )

                if training_utils.is_global_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * epoch_samples) + ((i + 1) * batch_size * args.world_size),
                    )

            # Update progress bar
            progress.update(n=batch_size * args.world_size)

        progress.close()

        # Epoch training metrics
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} training_loss: {running_loss.global_avg:.4f}")

        # Learning rate scheduler update
        if step_update is False:
            scheduler.step()
        if last_lr != float(max(scheduler.get_last_lr())):
            last_lr = float(max(scheduler.get_last_lr()))
            logger.info(f"Updated learning rate to: {last_lr}")

        # Checkpoint model
        if epoch % args.save_frequency == 0:
            training_utils.save_training_checkpoint(
                args,
                network_name,
                epoch,
                model_to_save,
                signature,
                {},
                rgb_stats,
                optimizer,
                scheduler,
                scaler,
                None,
            )
            training_utils.save_training_checkpoint(
                args,
                backbone_name,
                epoch,
                model_to_save.encoder,
                backbone_signature,
                {},
                rgb_stats,
                optimizer=None,
                scheduler=None,
                scaler=None,
                model_base=None,
            )
            if args.keep_last is not None and training_utils.is_global_primary(args) is True:
                fs_ops.clean_checkpoints(network_name, args.keep_last)
                fs_ops.clean_checkpoints(backbone_name, args.keep_last)

        # Epoch timing
        toc = time.time()
        logger.info(f"Total time: {format_duration(toc - tic)}")
        logger.info("---")

    summary_writer.close()

    # Checkpoint model
    training_utils.save_training_checkpoint(
        args,
        network_name,
        epoch,
        model_to_save,
        signature,
        {},
        rgb_stats,
        optimizer,
        scheduler,
        scaler,
        None,
    )
    training_utils.save_training_checkpoint(
        args,
        backbone_name,
        epoch,
        model_to_save.encoder,
        backbone_signature,
        {},
        rgb_stats,
        optimizer=None,
        scheduler=None,
        scaler=None,
        model_base=None,
    )

    training_utils.shutdown_distributed_mode(args)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Pretrain model",
        epilog=(
            "Usage examples\n"
            "==============\n"
            "torchrun --nproc_per_node=2 -m birder.scripts.train_eva \\\n"
            "    --network vit_b16 \\\n"
            "    --teacher vit_l16 \\\n"
            "    --batch-size 256 \\\n"
            "    --opt adamw --opt-betas 0.9 0.95 \\\n"
            "    --grad-accum-steps 8 \\\n"
            "    --lr 0.0003 \\\n"
            "    --lr-scale 256 \\\n"
            "    --wd 0.05 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --epochs 480 \\\n"
            "    --warmup-epochs 40 \\\n"
            "    --amp --amp-dtype bfloat16 \\\n"
            "    --compile \\\n"
            "    --data-path data/training\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the neural network to train")
    parser.add_argument("-t", "--tag", type=str, help="add model tag")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--teacher", type=str, help="the teacher network")
    parser.add_argument("--teacher-tag", type=str, help="teacher training log tag (loading only)")
    parser.add_argument(
        "--teacher-model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the teacher model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--teacher-epoch", type=int, help="load teacher weights from selected epoch")
    parser.add_argument("--mask-ratio", type=float, help="mask ratio for EVA training (default: model-specific)")
    parser.add_argument("--masking", type=str, choices=["block", "uniform"], default="block", help="masking strategy")
    parser.add_argument("--min-mask-size", type=int, default=1, help="minimum mask unit size in patches (uniform only)")
    training_cli.add_freeze_args(parser, scope_name="encoder")
    training_cli.add_optimization_args(parser)
    training_cli.add_lr_wd_args(parser)
    training_cli.add_lr_scheduler_args(parser)
    training_cli.add_training_schedule_args(parser, default_epochs=300)
    training_cli.add_batch_norm_args(parser)
    training_cli.add_input_args(
        parser, size_help="image size (defaults to teacher network size) shared by both networks"
    )
    training_cli.add_data_aug_args(parser, default_level=1, default_min_scale=0.25, default_re_prob=0.0)
    training_cli.add_dataloader_args(parser, default_drop_last=True)
    training_cli.add_precision_args(parser)
    training_cli.add_compile_args(parser, teacher=True)
    training_cli.add_checkpoint_args(parser)
    training_cli.add_distributed_args(parser)
    training_cli.add_logging_and_debug_args(parser, default_log_interval=100)
    training_cli.add_training_data_args(parser, unsupervised=True, wds_extra_shuffle=False)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = [str(p) for p in args.data_path]
    args.size = cli.parse_size(args.size)

    # This will capture the common argument mistakes
    training_cli.common_args_validation(args)

    # Script specific checks
    if args.teacher is None:
        raise cli.ValidationError("--teacher is required")
    if registry.exists(args.network, task=Task.IMAGE_CLASSIFICATION, net_type=MaskedTokenRetentionMixin) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")
    if registry.exists(args.teacher, task=Task.IMAGE_CLASSIFICATION) is False:
        raise cli.ValidationError(f"--teacher {args.teacher} not supported, see list-models tool for available options")
    if args.freeze_encoder_stages is not None and registry.exists(args.network, net_type=DetectorBackbone) is False:
        raise cli.ValidationError(
            "--freeze-encoder-stages only supported on detector backbone type networks, "
            "see list-models tool for available options"
        )


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    parser.set_defaults(**kwargs)
    args = parser.parse_args([])
    validate_args(args)

    return args


def main(overrides: Optional[TrainOverrides] = None) -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.wds_cache_dir is not None and Path(args.wds_cache_dir).exists() is False:
        logger.info(f"Creating {args.wds_cache_dir} directory...")
        Path(args.wds_cache_dir).mkdir(parents=True, exist_ok=True)

    train(args, overrides=overrides)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
