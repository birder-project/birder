"""
DINO v1, adapted from
https://github.com/facebookresearch/dino/blob/main/main_dino.py

Paper "Emerging Properties in Self-Supervised Vision Transformers", https://arxiv.org/abs/2104.14294

Changes from original:
* Per epoch weight decay scheduling (instead of per step)
"""

# Reference license: Apache-2.0

import argparse
import json
import logging
import os
import sys
import time
import typing
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.amp
import torch.utils.data
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.folder import pil_loader  # Slower but Handles external dataset quirks better
from torchvision.io import ImageReadMode
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import training_utils
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.dataloader.webdataset import make_wds_loader
from birder.datasets.directory import make_image_dataset
from birder.datasets.webdataset import make_wds_dataset
from birder.datasets.webdataset import prepare_wds_args
from birder.datasets.webdataset import wds_args_from_info
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import get_signature
from birder.net.ssl.dino_v1 import DINO_v1
from birder.net.ssl.dino_v1 import DINOHead
from birder.net.ssl.dino_v1 import DINOLoss
from birder.transforms.classification import RGBMode
from birder.transforms.classification import RGBType
from birder.transforms.classification import get_rgb_stats
from birder.transforms.classification import training_preset

logger = logging.getLogger(__name__)


def _tv_loader(path: str) -> torch.Tensor:
    if path.endswith(".webp") is True:  # Memory leak in TV webp
        return pil_to_tensor(pil_loader(path))

    return decode_image(path, mode=ImageReadMode.RGB)


class TrainTransform:
    def __init__(
        self,
        size: tuple[int, int],
        crop_size: tuple[int, int],
        level: int,
        rgv_values: RGBType,
        local_crops_number: int,
        resize_min_scale: Optional[float] = None,
    ) -> None:
        self.local_crops_number = local_crops_number
        self.global_transform = training_preset(size, level, rgv_values, resize_min_scale)

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

    def __call__(self, image: Any) -> list[torch.Tensor]:
        crops = []
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))

        return crops


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
    training_transform = TrainTransform(
        args.size, args.local_crop_size, args.aug_level, rgb_stats, args.local_crops_number, args.resize_min_scale
    )
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info_file is not None:
            (wds_path, dataset_size) = wds_args_from_info(args.wds_info_file, args.wds_split)
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
            cache_dir=args.wds_cache_dir,
        )

    else:
        training_dataset = make_image_dataset(
            args.data_path,
            {},
            transforms=training_transform,
            loader=pil_loader if args.img_loader == "pil" else _tv_loader,
        )

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training on {len(training_dataset):,} samples")

    batch_size: int = args.batch_size

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
            drop_last=True,
        )

    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=True,
            drop_last=True,
        )

    last_batch_idx = (len(training_dataset) // batch_size) - 1  # no partial batches
    begin_epoch = 1
    epochs = args.epochs + 1
    if args.stop_epoch is None:
        args.stop_epoch = epochs
    else:
        args.stop_epoch += 1

    #
    # Initialize networks
    #
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    sample_shape = (batch_size, args.channels, *args.size)  # B, C, H, W
    backbone_name = get_network_name(args.network, net_param=args.net_param, tag="dino-v1")
    network_name = get_mim_network_name(
        "dino_v1",
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
    if args.model_config is not None:
        teacher_model_config = args.model_config.copy()
        teacher_model_config.update({"drop_path_rate": 0.0})
    else:
        teacher_model_config = {"drop_path_rate": 0.0}

    teacher_backbone = registry.net_factory(
        args.network,
        sample_shape[1],
        0,
        net_param=args.net_param,
        config=teacher_model_config,
        size=args.size,
    )
    student_backbone.set_dynamic_size()
    teacher_backbone.set_dynamic_size()
    student_dino_head = DINOHead(
        student_backbone.embedding_size,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        num_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    )
    teacher_dino_head = DINOHead(
        teacher_backbone.embedding_size,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=True,
        num_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    )
    student = DINO_v1(student_backbone.input_channels, student_backbone, student_dino_head)
    teacher = DINO_v1(teacher_backbone.input_channels, teacher_backbone, teacher_dino_head)
    teacher.load_state_dict(student.state_dict())

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # 2 global crops + local crops number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=0.1,
        center_momentum=0.9,
    )

    net = torch.nn.ModuleDict(
        {
            "student": student,
            "teacher": teacher,
            "loss": dino_loss,
        }
    )
    net.task = teacher.task

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, training_states) = fs_ops.load_simple_checkpoint(device, net, network_name, epoch=args.resume_epoch)
        student = net["student"]
        teacher = net["teacher"]
        dino_loss = net["loss"]

    else:
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
    if args.sync_bn is True and args.distributed is True:
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

    # Teacher momentum and weight decay schedule
    momentum_schedule = training_utils.cosine_scheduler(args.momentum_teacher, 1.0, args.epochs, last_batch_idx)
    if args.wd_end is not None:
        wd_schedule = training_utils.cosine_scheduler(args.wd, args.wd_end, args.epochs, 1)
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
    # Distributed (DDP)
    #
    student_without_ddp = student
    teacher_without_ddp = teacher
    if args.distributed is True:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        student_without_ddp = student.module
        teacher_without_ddp = teacher.module

    # There is no backpropagation through the teacher
    for p in teacher.parameters():
        p.requires_grad = False

    model_to_save = net
    if teacher_compile_flag is True and hasattr(model_to_save["teacher"], "_orig_mod") is True:
        model_to_save["teacher"] = model_to_save["teacher"]._orig_mod  # pylint: disable=protected-access
    if args.compile is True and hasattr(model_to_save["student"], "_orig_mod") is True:
        model_to_save["student"] = model_to_save["student"]._orig_mod  # pylint: disable=protected-access

    #
    # Misc
    #

    # Print network summary
    net_for_info = teacher_without_ddp
    if teacher_compile_flag is True and hasattr(teacher_without_ddp, "_orig_mod") is True:
        net_for_info = teacher_without_ddp._orig_mod  # pylint: disable=protected-access

    if args.no_summary is False:
        torchinfo.summary(
            net_for_info,
            device=device,
            input_data={"xs": [torch.rand(sample_shape), torch.rand(sample_shape)]},
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

    signature = get_signature(input_shape=sample_shape, num_outputs=0)
    if args.rank == 0:
        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
        with open(training_log_path.joinpath("args.json"), "w", encoding="utf-8") as handle:
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

        if args.distributed is True:
            train_sampler.set_epoch(epoch)

        if wd_schedule is not None:
            wd = wd_schedule[epoch - 1]
            for param_group in optimizer.param_groups:
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd

            logger.info(f"Updated wd to: {wd}")

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

        for i, (_, images, _) in enumerate(training_loader):
            global_step = ((epoch - 1) * last_batch_idx) + i
            images = [img.to(device, dtype=model_dtype, non_blocking=True) for img in images]

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                teacher_output = teacher(images[:2])  # Only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch - 1)

            if args.freeze_last_layer_epochs >= epoch:
                student_without_ddp.head.cancel_last_layer_gradients()

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

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[global_step]
                for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

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

        # Learning rate scheduler update
        scheduler.step()
        if last_lr != max(scheduler.get_last_lr()):
            last_lr = max(scheduler.get_last_lr())
            logger.info(f"Updated learning rate to: {last_lr}")

        if args.rank == 0:
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
                    signature,
                    {},
                    rgb_stats,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    model_base=None,
                )

        # Epoch timing
        toc = time.time()
        (minutes, seconds) = divmod(toc - tic, 60)
        logger.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logger.info("---")

    summary_writer.close()

    # Checkpoint model
    if args.distributed is False or (args.distributed is True and args.rank == 0):
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
            signature,
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
            "torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 \\\n"
            "    --network xcit_small12_p16 \\\n"
            "    --local-crops-number 10 \\\n"
            "    --teacher-temp 0.07 \\\n"
            "    --opt adamw \\\n"
            "    --lr 0.00025 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --lr-cosine-min 1e-6 \\\n"
            "    --epochs 300 \\\n"
            "    --warmup-epochs 10 \\\n"
            "    --batch-size 128 \\\n"
            "    --wd 0.04 \\\n"
            "    --wd-end 0.4 \\\n"
            "    --norm-wd 0 \\\n"
            "    --bias-weight-decay 0 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
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
    parser.add_argument("--out-dim", type=int, default=65536, help="dimensionality of the DINO head output")
    parser.add_argument(
        "--use-bn-in-head",
        default=False,
        action="store_true",
        help="whether to use batch normalizations in projection head",
    )
    parser.add_argument(
        "--norm-last-layer",
        default=False,
        action="store_true",
        help=(
            "whether or not to weight normalize the last layer of the DINO head, "
            "set this flag with large models (vit_b)"
        ),
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
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument("--compile-teacher", default=False, action="store_true", help="enable teacher only compilation")
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
    parser.add_argument("--wd-end", type=float, help="final value of the weight decay (None for constant wd)")
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
    parser.add_argument("--sync-bn", default=False, action="store_true", help="use synchronized BatchNorm")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N", help="number of warmup epochs")
    parser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=2,
        help="magnitude of augmentations (0 off -> 4 highest)",
    )
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--resize-min-scale", type=float, default=0.3, help="random resize min scale")
    parser.add_argument("--epochs", type=int, default=400, metavar="N", help="number of training epochs")
    parser.add_argument(
        "--stop-epoch", type=int, metavar="N", help="epoch to stop the training at (multi step training)"
    )
    parser.add_argument("--save-frequency", type=int, default=1, metavar="N", help="frequency of model saving")
    parser.add_argument("--resume-epoch", type=int, metavar="N", help="epoch to resume training from")
    parser.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
    parser.add_argument("--load-scheduler", default=False, action="store_true", help="load scheduler only resuming")
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
    parser.add_argument("--data-path", nargs="*", default=[], help="training directories paths (directories and files)")
    training_utils.add_unsupervised_wds_args(parser)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = [str(p) for p in args.data_path]
    assert args.network is not None
    assert args.load_states is False or (
        args.load_states is True and args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert (
        args.load_scheduler is False or args.resume_epoch is not None
    ), "Load scheduler must be from resumed training (--resume-epoch)"
    assert args.wds is True or len(args.data_path) >= 1
    assert args.wds is False or len(args.data_path) <= 1
    assert (
        registry.exists(args.network, task=Task.IMAGE_CLASSIFICATION) is True
    ), "Unknown network, see list-models tool for available options"
    assert args.amp is False or args.model_dtype == "float32"
    assert args.resize_min_scale is None or args.resize_min_scale < 1.0
    assert args.compile is False or args.compile_teacher is False
    args.size = cli.parse_size(args.size)
    args.local_crop_size = cli.parse_size(args.local_crop_size)


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
