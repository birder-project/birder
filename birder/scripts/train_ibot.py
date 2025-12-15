"""
iBOT, adapted from
https://github.com/bytedance/ibot/blob/main/main_ibot.py

Paper "iBOT: Image BERT Pre-Training with Online Tokenizer", https://arxiv.org/abs/2111.07832

Changes from original:
* Per epoch weight decay scheduling (instead of per step)
"""

# Reference license: Apache-2.0

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.folder import pil_loader  # Slower but Handles external dataset quirks better
from torchvision.transforms import v2
from tqdm import tqdm

import birder
from birder.common import cli
from birder.common import fs_ops
from birder.common import training_cli
from birder.common import training_utils
from birder.common.lib import format_duration
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.common.lib import set_random_seeds
from birder.common.masking import BlockMasking
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.directory import tv_loader
from birder.data.datasets.fake import FakeDataWithPaths
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import get_rgb_stats
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import get_signature
from birder.net.ssl.base import get_ssl_signature
from birder.net.ssl.ibot import iBOT
from birder.net.ssl.ibot import iBOTLoss

logger = logging.getLogger(__name__)


class TrainTransform:
    def __init__(
        self,
        global_transform: Callable[..., torch.Tensor],
        crop_size: tuple[int, int],
        rgv_values: RGBType,
        local_crops_number: int,
        mask_generator: Callable[[int], torch.Tensor],
    ) -> None:
        self.global_transform = global_transform
        self.local_crops_number = local_crops_number

        # Local small crops
        mean = rgv_values["mean"]
        std = rgv_values["std"]
        self.local_transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.RandomResizedCrop(crop_size, scale=(0.1, 0.5), interpolation=v2.InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.15, hue=0.04)], p=0.8),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1.2))], p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        self.mask_generator = mask_generator

    def __call__(self, image: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
        crops = []
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))

        masks = self.mask_generator(2)

        return (crops, masks)


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

    batch_size: int = args.batch_size
    logger.debug(f"Effective batch size = {args.batch_size * args.grad_accum_steps * args.world_size}")

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
    backbone_name = get_network_name(args.network, tag="ibot")
    if args.tag is not None:
        backbone_name = f"{backbone_name}-{args.tag}"

    network_name = get_mim_network_name("ibot", encoder=args.network, tag=args.tag)

    student_backbone = registry.net_factory(args.network, sample_shape[1], 0, config=args.model_config, size=args.size)
    if args.model_config is not None:
        teacher_model_config = args.model_config.copy()
        teacher_model_config.update({"drop_path_rate": 0.0})
    else:
        teacher_model_config = {"drop_path_rate": 0.0}

    teacher_backbone = registry.net_factory(
        args.network, sample_shape[1], 0, config=teacher_model_config, size=args.size
    )
    student_backbone.set_dynamic_size()
    student = iBOT(
        student_backbone,
        config={
            "out_dim": args.out_dim,
            "norm_last_layer": args.norm_last_layer,
            "num_layers": 3,
            "hidden_dim": 2048,
            "bottleneck_dim": 256,
            "patch_out_dim": args.patch_out_dim,
            "shared_head": args.shared_head,
        },
    )
    teacher = iBOT(
        teacher_backbone,
        config={
            "out_dim": args.out_dim,
            "norm_last_layer": True,
            "num_layers": 3,
            "hidden_dim": 2048,
            "bottleneck_dim": 256,
            "patch_out_dim": args.patch_out_dim,
            "shared_head": args.shared_head,
        },
    )
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    ibot_loss = iBOTLoss(
        args.out_dim,
        args.patch_out_dim,
        num_global_crops=2,
        num_local_crops=args.local_crops_number,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp2=args.warmup_teacher_temp,
        teacher_temp2=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        epochs=args.epochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    )

    net = torch.nn.ModuleDict(
        {
            "student": student,
            "teacher": teacher,
            "loss": ibot_loss,
        }
    )
    net.task = teacher.task

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, training_states) = fs_ops.load_simple_checkpoint(
            device, net, network_name, epoch=args.resume_epoch, strict=not args.non_strict_weights
        )
        student = net["student"]
        teacher = net["teacher"]
        ibot_loss = net["loss"]

    else:
        training_states = fs_ops.TrainingStates.empty()

    assert isinstance(student_backbone, MaskedTokenRetentionMixin)
    assert isinstance(net, torch.nn.Module)

    net.to(device, dtype=model_dtype)
    if args.freeze_bn is True:
        student = training_utils.freeze_batchnorm2d(student)
        teacher = training_utils.freeze_batchnorm2d(teacher)
    elif args.sync_bn is True and args.distributed is True:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile networks
    teacher_compile_flag = args.compile is True or args.compile_teacher is True
    if args.compile is True:
        student = torch.compile(student)
        teacher = torch.compile(teacher)
    elif args.compile_teacher is True:
        teacher = torch.compile(teacher)

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    logger.debug(f"Using RGB stats: {rgb_stats}")

    mask_size = (args.size[0] // student_backbone.max_stride, args.size[1] // student_backbone.max_stride)
    mask_generator = BlockMasking(
        mask_size,
        min_num_patches=4,
        max_num_patches=mask_size[0] * mask_size[1] // 2,
        min_aspect=0.33,
        max_aspect=3.33,
    )
    training_transform = TrainTransform(
        training_utils.get_training_transform(args),
        args.local_crop_size,
        rgb_stats,
        args.local_crops_number,
        mask_generator,
    )
    if args.use_fake_data is True:
        logger.warning("Using fake data")
        training_dataset = FakeDataWithPaths(
            10000, (args.channels, *args.size), num_classes=10, transform=training_transform
        )

    elif args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            (wds_path, dataset_size) = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_train_size is not None:
                dataset_size = args.wds_train_size
        else:
            (wds_path, dataset_size) = prepare_wds_args(args.data_path[0], args.wds_train_size, device)

        training_dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=True,
            transform=training_transform,
            img_loader=args.img_loader,
            cls_key=None,
            cache_dir=args.wds_cache_dir,
        )

    else:
        training_dataset = make_image_dataset(
            args.data_path,
            {},
            transforms=training_transform,
            loader=pil_loader if args.img_loader == "pil" else tv_loader,
        )

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training on {len(training_dataset):,} samples")

    # Data loaders and samplers
    if args.distributed is True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(training_dataset)

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
            world_size=args.world_size,
            pin_memory=True,
            drop_last=args.drop_last,
            shuffle=args.wds_extra_shuffle,
        )

    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=True,
            drop_last=args.drop_last,
        )

    last_batch_idx = len(training_loader) - 1

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

    # Teacher momentum and weight decay schedule
    momentum_schedule = training_utils.cosine_scheduler(args.momentum_teacher, 1.0, args.epochs, 0, last_batch_idx + 1)
    if args.wd_end is not None:
        wd_schedule = training_utils.cosine_scheduler(args.wd, args.wd_end, args.epochs, 0, 1)
    else:
        wd_schedule = None

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
    # Distributed (DDP)
    #

    # There is no backpropagation through the teacher
    for p in teacher.parameters():
        p.requires_grad = False

    student_without_ddp = student
    if args.distributed is True:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters
        )
        student_without_ddp = student.module

    model_to_save = net
    if teacher_compile_flag is True and hasattr(model_to_save["teacher"], "_orig_mod") is True:
        model_to_save["teacher"] = model_to_save["teacher"]._orig_mod  # pylint: disable=protected-access
    if args.compile is True and hasattr(model_to_save["student"], "_orig_mod") is True:
        model_to_save["student"] = model_to_save["student"]._orig_mod  # pylint: disable=protected-access

    #
    # Misc
    #

    # Print network summary
    net_for_info = teacher
    if teacher_compile_flag is True and hasattr(teacher, "_orig_mod") is True:
        net_for_info = teacher._orig_mod  # pylint: disable=protected-access

    if args.no_summary is False:
        summary = torchinfo.summary(
            net_for_info,
            device=device,
            input_data={"x": torch.rand(sample_shape), "masks": mask_generator(batch_size)},
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

    signature = get_ssl_signature(input_shape=sample_shape)
    backbone_signature = get_signature(input_shape=sample_shape, num_outputs=0)
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
                {"training_samples": len(training_dataset)},
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
        train_accuracy = training_utils.SmoothedValue()

        if args.distributed is True:
            train_sampler.set_epoch(epoch)

        if wd_schedule is not None:
            wd = wd_schedule[epoch - 1]
            for param_group in optimizer.param_groups:
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd

            logger.info(f"Updated wd to: {wd}")

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
        for i, (_, (images, masks), _) in enumerate(training_loader):
            global_step = ((epoch - 1) * (last_batch_idx + 1)) + i
            images = [img.to(device, dtype=model_dtype, non_blocking=True) for img in images]
            if args.pred_start_epoch >= epoch:
                masks = None
            else:
                masks = masks.to(device, dtype=model_dtype, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                # Global views
                (teacher_embedding, teacher_features) = teacher(torch.concat(images[:2], dim=0), None)
                (student_embedding, student_features) = student(torch.concat(images[:2], dim=0), masks)

                # Local views
                (student_local_embedding, _) = student(torch.concat(images[2:], dim=0), None, return_keys="embedding")

                loss = ibot_loss(
                    student_embedding,
                    student_features,
                    teacher_embedding,
                    teacher_features,
                    student_local_embedding,
                    masks.reshape(batch_size * 2, -1),
                    epoch - 1,
                )["all"]

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.freeze_last_layer_epochs >= epoch:
                    student_without_ddp.head.cancel_last_layer_gradients()

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
                if args.freeze_last_layer_epochs >= epoch:
                    student_without_ddp.head.cancel_last_layer_gradients()

                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                    if iter_update is True:
                        scheduler.step()

            if optimizer_update is True:
                # EMA update for the teacher
                with torch.no_grad():
                    m = momentum_schedule[global_step]
                    torch._foreach_lerp_(  # pylint: disable=protected-access
                        list(teacher.parameters()), list(student.parameters()), weight=1 - m
                    )

            # Statistics
            running_loss.update(loss.detach())

            probs_teacher = teacher_embedding.chunk(2)  # Per global crop
            probs_student = student_embedding.chunk(2)
            pred_teacher = probs_teacher[0].argmax(dim=1)
            pred_student = probs_student[1].argmax(dim=1)
            train_accuracy.update(training_utils.accuracy(pred_teacher, pred_student))

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

                if training_utils.is_local_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )
                    summary_writer.add_scalars(
                        "performance",
                        {"accuracy": train_accuracy.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )

            # Update progress bar
            progress.update(n=batch_size * args.world_size)

        progress.close()

        # Epoch training metrics
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} training_loss: {running_loss.global_avg:.4f}")
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} accuracy: {train_accuracy.global_avg:.4f}")

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
                    {},
                    rgb_stats,
                    optimizer,
                    scheduler,
                    scaler,
                    None,
                )
                fs_ops.checkpoint_model(
                    backbone_name,
                    epoch,
                    model_to_save["teacher"].backbone,
                    backbone_signature,
                    {},
                    rgb_stats,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    model_base=None,
                )
                if args.keep_last is not None:
                    fs_ops.clean_checkpoints(network_name, args.keep_last)
                    fs_ops.clean_checkpoints(backbone_name, args.keep_last)

        # Epoch timing
        toc = time.time()
        logger.info(f"Total time: {format_duration(toc - tic)}")
        logger.info("---")

        # Reset counters
        epoch_start = time.time()
        start_time = epoch_start
        last_idx = 0

    summary_writer.close()

    # Checkpoint model
    if training_utils.is_local_primary(args) is True:
        fs_ops.checkpoint_model(
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
        fs_ops.checkpoint_model(
            backbone_name,
            epoch,
            model_to_save["teacher"].backbone,
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
        description="Pre-train model",
        epilog=(
            "Usage examples\n"
            "==============\n"
            "torchrun --nproc_per_node=2 -m birder.scripts.train_ibot \\\n"
            "    --network vit_b16 \\\n"
            "    --shared-head \\\n"
            "    --norm-last-layer \\\n"
            "    --local-crops-number 10 \\\n"
            "    --teacher-temp 0.07 \\\n"
            "    --warmup-teacher-temp-epochs 50 \\\n"
            "    --opt adamw \\\n"
            "    --lr 0.00075 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --lr-cosine-min 1e-6 \\\n"
            "    --freeze-last-layer-epochs 3 \\\n"
            "    --epochs 400 \\\n"
            "    --warmup-epochs 10 \\\n"
            "    --batch-size 32 \\\n"
            "    --wd 0.04 \\\n"
            "    --wd-end 0.4 \\\n"
            "    --norm-wd 0 \\\n"
            "    --bias-weight-decay 0 \\\n"
            "    --clip-grad-norm 0.3 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
            "    --data-path data/training\n"
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
    parser.add_argument("--out-dim", type=int, default=8192, help="dimensionality of embedding output")
    parser.add_argument("--patch-out-dim", type=int, default=8192, help="dimensionality of features output")
    parser.add_argument(
        "--shared-head",
        default=False,
        action="store_true",
        help="wether to share the same head for embedding and features outputs (if false patch_out_dim is ignored)",
    )
    parser.add_argument(
        "--norm-last-layer",
        default=False,
        action="store_true",
        help="whether or not to weight normalize the last layer of the head, set this flag with large models (vit_b)",
    )
    parser.add_argument(
        "--momentum-teacher",
        type=float,
        default=0.996,
        help="base EMA parameter for teacher update, set a higher value with small batches",
    )
    parser.add_argument("--local-crops-number", type=int, default=8, help="number of small local views to generate")
    parser.add_argument(
        "--local-crop-size", type=int, nargs="+", default=[96, 96], metavar=("H", "W"), help="local view size"
    )
    parser.add_argument(
        "--pred-start-epoch", type=int, default=0, help="start epoch to perform masked image prediction"
    )
    parser.add_argument("--lambda1", default=1.0, type=float, help="loss weight for dino loss over embedding")
    parser.add_argument("--lambda2", default=1.0, type=float, help="loss weight for beit loss over features")
    parser.add_argument(
        "--warmup-teacher-temp",
        type=float,
        default=0.04,
        help="initial value for the teacher temperature, try decreasing it if the training loss does not decrease",
    )
    parser.add_argument(
        "--teacher-temp", type=float, default=0.04, help="final value (after linear warmup) of the teacher temperature"
    )
    parser.add_argument(
        "--warmup-teacher-temp-epochs", type=int, default=40, help="number of warmup epochs for the teacher temperature"
    )
    parser.add_argument(
        "--freeze-last-layer-epochs",
        default=1,
        type=int,
        help=(
            "number of epochs during which the output layer is frozen, "
            "try increasing this value if the loss does not decrease"
        ),
    )
    parser.add_argument("-t", "--tag", type=str, help="add model tag")
    training_cli.add_optimization_args(parser)
    training_cli.add_lr_wd_args(parser, wd_end=True)
    training_cli.add_lr_scheduler_args(parser)
    training_cli.add_training_schedule_args(parser, default_epochs=800)
    training_cli.add_input_args(parser)
    training_cli.add_data_aug_args(parser, default_level=1, default_min_scale=0.3, default_re_prob=0.0)
    training_cli.add_dataloader_args(parser, default_drop_last=True)
    training_cli.add_batch_norm_args(parser)
    training_cli.add_precision_args(parser)
    training_cli.add_compile_args(parser, teacher=True)
    training_cli.add_checkpoint_args(parser)
    training_cli.add_distributed_args(parser)
    training_cli.add_logging_and_debug_args(parser, default_log_interval=100)
    training_cli.add_training_data_args(parser, unsupervised=True)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = [str(p) for p in args.data_path]
    args.size = cli.parse_size(args.size)
    args.local_crop_size = cli.parse_size(args.local_crop_size)

    # This will capture the common argument mistakes
    training_cli.common_args_validation(args)

    # Script specific checks
    if registry.exists(args.network, task=Task.IMAGE_CLASSIFICATION, net_type=MaskedTokenRetentionMixin) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")


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
