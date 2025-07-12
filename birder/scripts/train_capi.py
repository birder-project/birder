"""
CAPI, adapted from
https://github.com/facebookresearch/capi/blob/main/train_capi.py

Paper "Cluster and Predict Latent Patches for Improved Masked Image Modeling",
https://arxiv.org/abs/2502.08769

Changes from original:
* No LR truncation
"""

# Reference license: Apache-2.0

import argparse
import json
import logging
import math
import os
import sys
import time
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.folder import pil_loader  # Slower but Handles external dataset quirks better
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import masking
from birder.common import training_utils
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.directory import tv_loader
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import RGBMode
from birder.data.transforms.classification import get_rgb_stats
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import get_signature
from birder.net.ssl.base import get_ssl_signature
from birder.net.ssl.capi import CAPIStudent
from birder.net.ssl.capi import CAPITeacher

logger = logging.getLogger(__name__)


class TrainCollator:
    def __init__(self, mask_generator: Callable[[int], tuple[list[torch.Tensor], list[torch.Tensor]]]) -> None:
        self.mask_generator = mask_generator

    def __call__(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        masks = self.mask_generator(B)

        return (collated_batch, masks)


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

    batch_size: int = args.batch_size
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
    backbone_name = get_network_name(args.network, net_param=args.net_param, tag="capi")
    network_name = get_mim_network_name(
        "capi",
        net_param=None,
        encoder=args.network,
        encoder_param=args.net_param,
        tag=args.tag,
    )

    student_backbone = registry.net_factory(
        args.network,
        sample_shape[1],
        0,
        net_param=args.net_param,
        config=args.model_config,
        size=args.size,
    )
    teacher_backbone = registry.net_factory(
        args.network,
        sample_shape[1],
        0,
        net_param=args.net_param,
        config=args.model_config,
        size=args.size,
    )

    teacher_backbone.load_state_dict(student_backbone.state_dict())

    student = CAPIStudent(
        student_backbone,
        config={
            "decoder_layers": args.decoder_layers,
            "decoder_dim": args.decoder_dim,
            "num_clusters": args.num_clusters,
        },
    )
    teacher = CAPITeacher(
        teacher_backbone,
        config={
            "num_clusters": args.num_clusters,
            "bias": True,
            "n_sk_iter": 3,
            "target_temp": 0.06,
            "pred_temp": 0.12,
        },
    )

    net = torch.nn.ModuleDict(
        {
            "student": student,
            "teacher": teacher,
        }
    )
    net.task = student.task

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, training_states) = fs_ops.load_simple_checkpoint(device, net, network_name, epoch=args.resume_epoch)
        student = net["student"]
        teacher = net["teacher"]

    else:
        training_states = fs_ops.TrainingStates.empty()

    teacher.eval()

    assert isinstance(student_backbone, MaskedTokenOmissionMixin)
    assert isinstance(net, torch.nn.Module)

    net.to(device, dtype=model_dtype)
    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile networks
    if args.compile is True:
        student = torch.compile(student)
        teacher = torch.compile(teacher)

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode)
    mask_size = (args.size[0] // student_backbone.max_stride, args.size[1] // student_backbone.max_stride)
    seq_len = mask_size[0] * mask_size[1]
    mask_generator = masking.InverseRollBlockMasking(
        mask_size,
        num_masking_patches=int(seq_len * args.mask_ratio),
        min_aspect=0.5,
        max_aspect=2.0,
    )
    n_masked = int(seq_len * args.mask_ratio)
    n_predict = int(n_masked * args.kept_mask_ratio)
    mask_collator = TrainCollator(mask_generator)
    training_transform = training_utils.get_training_transform(args)
    if args.wds is True:
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
            collate_fn=mask_collator,
            world_size=args.world_size,
            pin_memory=True,
            drop_last=True,
        )

    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=mask_collator,
            pin_memory=True,
            drop_last=True,
        )

    last_batch_idx = len(training_loader) - 1

    #
    # Loss criteria, optimizer, learning rate scheduler and training parameter groups
    #

    # Training parameter groups
    custom_keys_weight_decay = training_utils.get_wd_custom_keys(args)
    parameters = training_utils.optimizer_parameter_groups(
        student,
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
    clustering_lr = lr / 2
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
    clustering_optimizer = torch.optim.AdamW(teacher.head.parameters(), lr=clustering_lr, betas=[0.9, 0.95])
    scheduler = training_utils.get_scheduler(optimizer, iters_per_epoch, args)
    clustering_scheduler = training_utils.get_scheduler(clustering_optimizer, iters_per_epoch, args)
    if args.compile_opt is True:
        optimizer.step = torch.compile(optimizer.step, fullgraph=False)
        clustering_optimizer.step = torch.compile(clustering_optimizer.step, fullgraph=False)

    # Momentum and temperatures
    momentum_schedule = training_utils.cosine_scheduler(
        args.momentum_teacher, 1.0, args.epochs, args.warmup_epochs, last_batch_idx + 1, start_warmup_value=1.0
    )
    student_temp = 0.12

    # Gradient scaler and AMP related tasks
    (scaler, amp_dtype) = training_utils.get_amp_scaler(args.amp, args.amp_dtype)
    (clustering_scaler, _) = training_utils.get_amp_scaler(args.amp, args.amp_dtype)

    # Load states
    if args.load_states is True:
        optimizer.load_state_dict(training_states.optimizer_state)
        scheduler.load_state_dict(training_states.scheduler_state)
        clustering_optimizer.load_state_dict(training_states.extra_states["clustering_optimizer"])  # type: ignore
        clustering_scheduler.load_state_dict(training_states.extra_states["clustering_scheduler"])  # type: ignore
        if scaler is not None:
            scaler.load_state_dict(training_states.scaler_state)
            clustering_scaler.load_state_dict(training_states.extra_states["clustering_scaler"])  # type: ignore

    last_lr = max(scheduler.get_last_lr())
    if args.plot_lr is True:
        logger.info("Fast forwarding scheduler...")
        lrs = []
        for _ in range(begin_epoch, epochs):
            for _ in range(iters_per_epoch):
                optimizer.step()
                lrs.append(max(scheduler.get_last_lr()))
                scheduler.step()

        plt.plot(np.linspace(begin_epoch, epochs, iters_per_epoch * (epochs - begin_epoch), endpoint=False), lrs)
        plt.show()
        raise SystemExit(0)

    #
    # Distributed (DDP)
    #

    # There is no backpropagation through the teacher backbone
    for p in teacher.backbone.parameters():
        p.requires_grad = False

    teacher_without_ddp = teacher
    student_without_ddp = student
    if args.distributed is True:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters
        )
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank])
        student_without_ddp = student.module
        teacher_without_ddp = teacher.module

    model_to_save = net
    if args.compile is True and hasattr(model_to_save["teacher"], "_orig_mod") is True:
        model_to_save["teacher"] = model_to_save["teacher"]._orig_mod  # pylint: disable=protected-access
    if args.compile is True and hasattr(model_to_save["student"], "_orig_mod") is True:
        model_to_save["student"] = model_to_save["student"]._orig_mod  # pylint: disable=protected-access

    #
    # Misc
    #

    # Print network summary
    net_for_info = student_without_ddp
    if args.compile is True and hasattr(student_without_ddp, "_orig_mod") is True:
        net_for_info = student_without_ddp._orig_mod  # pylint: disable=protected-access

    if args.no_summary is False:
        all_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        torchinfo.summary(
            net_for_info,
            device=device,
            input_data=(
                torch.rand(sample_shape),
                all_ids.repeat(batch_size, 1),
                all_ids.repeat(batch_size, 1)[:, : mask_size[0]],
            ),
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

    signature = get_ssl_signature(input_shape=sample_shape)
    backbone_signature = get_signature(input_shape=sample_shape, num_outputs=0)
    if training_utils.is_local_primary(args) is True:
        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
        with open(training_log_path.joinpath("training_args.json"), "w", encoding="utf-8") as handle:
            json.dump({"cmdline": " ".join(sys.argv), **vars(args)}, handle, indent=2)

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
        running_clustering_loss = training_utils.SmoothedValue()
        running_target_entropy = training_utils.SmoothedValue()

        if args.distributed is True:
            train_sampler.set_epoch(epoch)

        if training_utils.is_local_primary(args) is True:
            progress = tqdm(
                desc=f"Epoch {epoch}/{epochs-1}",
                total=len(training_dataset),
                initial=0,
                unit="samples",
                leave=False,
            )

        # Zero the parameter gradients
        optimizer.zero_grad()
        clustering_optimizer.zero_grad()

        for i, ((_, images, _), masks) in enumerate(training_loader):
            global_step = ((epoch - 1) * (last_batch_idx + 1)) + i
            images = images.to(device, dtype=model_dtype, non_blocking=True)
            masks = masks.to(device, dtype=model_dtype, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Mask handling
            ids_keep = masking.get_ids_keep(masks)
            predict_indices = masking.get_random_masked_indices(masks, n_predict)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                (selected_assignments, clustering_loss) = teacher(images, None, predict_indices)

            if clustering_scaler is not None:
                clustering_scaler.scale(clustering_loss).backward()
                if optimizer_update is True:
                    clustering_scaler.step(clustering_optimizer)
                    clustering_scaler.update()
                    clustering_optimizer.zero_grad()
                    if iter_update is True:
                        clustering_scheduler.step()

            else:
                clustering_loss.backward()
                if optimizer_update is True:
                    clustering_optimizer.step()
                    clustering_optimizer.zero_grad()
                    if iter_update is True:
                        clustering_scheduler.step()

            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                pred = student(images, ids_keep, predict_indices)
                loss = -torch.sum(selected_assignments * F.log_softmax(pred / student_temp, dim=-1), dim=-1)
                target_entropy = -torch.xlogy(selected_assignments, selected_assignments).sum(dim=-1).mean()

            loss = loss.sum() / len(loss)

            if scaler is not None:
                scaler.scale(loss).backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if iter_update is True:
                        scheduler.step()

            else:
                loss.backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                    if iter_update is True:
                        scheduler.step()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[global_step]
                for param_q, param_k in zip(
                    student_without_ddp.backbone.parameters(), teacher_without_ddp.backbone.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Statistics
            running_loss.update(loss.detach())
            running_clustering_loss.update(clustering_loss.detach())
            running_target_entropy.update(target_entropy.detach())

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                running_loss.synchronize_between_processes(device)
                running_clustering_loss.synchronize_between_processes(device)
                running_target_entropy.synchronize_between_processes(device)
                if training_utils.is_local_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {
                            "training": running_loss.avg,
                            "clustering": running_clustering_loss.avg,
                        },
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )
                    summary_writer.add_scalars(
                        "performance",
                        {"target_entropy": running_target_entropy.avg},
                        ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                    )

            # Update progress bar
            if training_utils.is_local_primary(args) is True:
                progress.update(n=batch_size * args.world_size)

        if training_utils.is_local_primary(args) is True:
            progress.close()

        # Epoch training metrics
        epoch_loss = running_loss.global_avg
        logger.info(f"Epoch {epoch}/{epochs-1} training_loss: {epoch_loss:.4f}")

        epoch_clustering_loss = running_clustering_loss.global_avg
        logger.info(f"Epoch {epoch}/{epochs-1} clustering_loss: {epoch_clustering_loss:.4f}")

        epoch_target_entropy = running_target_entropy.global_avg
        logger.info(f"Epoch {epoch}/{epochs-1} target_entropy: {epoch_target_entropy:.4f}")

        # Learning rate scheduler update
        if iter_update is False:
            scheduler.step()
            clustering_scheduler.step()
        if last_lr != max(scheduler.get_last_lr()):
            last_lr = max(scheduler.get_last_lr())
            logger.info(f"Updated learning rate to: {last_lr}")

        if training_utils.is_local_primary(args) is True:
            extra_states = {
                "clustering_optimizer": clustering_optimizer.state_dict(),
                "clustering_scheduler": clustering_scheduler.state_dict(),
            }
            if clustering_scaler is not None:
                extra_states.update({"clustering_scaler": clustering_scaler.state_dict()})

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
                    **extra_states,
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
        (minutes, seconds) = divmod(toc - tic, 60)
        logger.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logger.info("---")

    summary_writer.close()

    # Checkpoint model
    if training_utils.is_local_primary(args) is True:
        extra_states = {
            "clustering_optimizer": clustering_optimizer.state_dict(),
            "clustering_scheduler": clustering_scheduler.state_dict(),
        }
        if clustering_scaler is not None:
            extra_states.update({"clustering_scaler": clustering_scaler.state_dict()})

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
            **extra_states,
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
            "torchrun --nproc_per_node=2 -m birder.scripts.train_capi \\\n"
            "    --network rope_vit_reg4_s14 \\\n"
            "    --opt adamw \\\n"
            "    --lr 0.001 \\\n"
            "    --opt-betas 0.9 0.95 \\\n"
            "    --lr-scheduler-update iter \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --lr-cosine-min 1e-7 \\\n"
            "    --warmup-epochs 40 \\\n"
            "    --batch-size 256 \\\n"
            "    --epochs 400 \\\n"
            "    --wd 0.1 \\\n"
            "    --norm-wd 0.01 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
            "    --compile-opt \\\n"
            "    --data-path data/training\n"
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
    parser.add_argument("--decoder-layers", type=int, default=12, help="number of decoder layers")
    parser.add_argument("--decoder-dim", type=int, default=1024, help="decoder dimensionality")
    parser.add_argument("--num-clusters", type=int, default=16384, help="clustering head width")
    parser.add_argument("--mask-ratio", type=float, default=0.65, help="masking ratio")
    parser.add_argument("--kept-mask-ratio", type=float, default=0.05, help="subsampling ratio for decoding")
    parser.add_argument("--momentum-teacher", type=float, default=0.999, help="base EMA parameter for teacher update")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-opt", default=False, action="store_true", help="enable compilation for optimizer step"
    )
    training_utils.add_optimizer_args(parser)
    training_utils.add_lr_wd_args(parser)
    training_utils.add_scheduler_args(parser)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, metavar="N", help="number of steps to accumulate gradients"
    )
    parser.add_argument("--channels", type=int, default=3, metavar="N", help="no. of image channels")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size (defaults to network recommendation)"
    )
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N", help="number of warmup epochs")
    training_utils.add_aug_args(parser, default_level=1, default_min_scale=0.6)
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--epochs", type=int, default=200, metavar="N", help="number of training epochs")
    parser.add_argument(
        "--stop-epoch", type=int, metavar="N", help="epoch to stop the training at (multi step training)"
    )
    parser.add_argument("--save-frequency", type=int, default=1, metavar="N", help="frequency of model saving")
    parser.add_argument("--keep-last", type=int, metavar="N", help="number of checkpoints to keep")
    parser.add_argument("--resume-epoch", type=int, metavar="N", help="epoch to resume training from")
    parser.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
    parser.add_argument("-t", "--tag", type=str, help="add training logs tag")
    parser.add_argument(
        "--log-interval", type=int, default=50, metavar="N", help="how many steps between summary writes"
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
        "--img-loader", type=str, choices=["tv", "pil"], default="tv", help="backend to load and decode images"
    )
    parser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
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
    parser.add_argument(
        "--find-unused-parameters",
        default=False,
        action="store_true",
        help="enable searching for unused parameters in DistributedDataParallel (may impact performance)",
    )
    parser.add_argument("--clip-grad-norm", type=float, help="the maximum gradient norm")
    parser.add_argument("--local-rank", type=int, help="local rank")
    parser.add_argument("--cpu", default=False, action="store_true", help="use cpu (mostly for testing)")
    parser.add_argument(
        "--use-deterministic-algorithms", default=False, action="store_true", help="use only deterministic algorithms"
    )
    parser.add_argument(
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
    parser.add_argument("--no-summary", default=False, action="store_true", help="don't print model summary")
    parser.add_argument("--data-path", nargs="*", default=[], help="training directories paths (directories and files)")
    training_utils.add_unsupervised_wds_args(parser)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = [str(p) for p in args.data_path]
    assert args.network is not None
    assert args.load_states is False or (
        args.load_states is True and args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert args.wds is True or len(args.data_path) >= 1
    assert args.wds is False or len(args.data_path) <= 1
    assert (
        registry.exists(args.network, task=Task.IMAGE_CLASSIFICATION) is True
    ), "Unknown network, see list-models tool for available options"
    assert args.amp is False or args.model_dtype == "float32"
    assert args.resize_min_scale is None or args.resize_min_scale < 1.0
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
