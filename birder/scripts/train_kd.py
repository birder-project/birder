"""
Knowledge Distillation training script.
Supports:
 * Logits matching (Soft distillation), https://arxiv.org/abs/1503.02531
 * Hard-label distillation, https://arxiv.org/pdf/2012.12877
 * Distillation token, https://arxiv.org/pdf/2012.12877
"""

import argparse
import json
import logging
import math
import sys
import time
import typing
from pathlib import Path
from typing import Any
from typing import Literal

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
from torchvision.io import read_image
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common import training_utils
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.dataloader.webdataset import make_wds_loader
from birder.datasets.webdataset import make_wds_dataset
from birder.datasets.webdataset import wds_size
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import get_signature
from birder.transforms.classification import get_mixup_cutmix
from birder.transforms.classification import inference_preset
from birder.transforms.classification import training_preset

DistType = Literal["soft", "hard", "deit"]


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    distillation_type: DistType = args.type

    training_utils.init_distributed_mode(args)
    if args.aa is True:
        args.aug_level = -1

    if args.type != "soft":
        args.temperature = 1.0

    logging.info(f"Using size={args.size}")

    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    torch.backends.cudnn.benchmark = True

    # Using the teacher rgb values for the student
    (teacher, class_to_idx, signature, rgb_values) = fs_ops.load_model(
        device,
        args.teacher,
        net_param=args.teacher_param,
        tag=args.teacher_tag,
        epoch=args.teacher_epoch,
        new_size=args.size,
        inference=True,
        pts=args.pts,
        pt2=args.pt2,
    )
    if args.size is None:
        args.size = lib.get_size_from_signature(signature)[0]
        logging.debug(f"Using size={args.size}")

    if args.wds is True:
        (wds_path, _) = fs_ops.wds_braces_from_path(Path(args.data_path))
        if args.wds_train_size is not None:
            dataset_size = args.wds_train_size

        else:
            dataset_size = wds_size(wds_path, device)

        training_dataset = make_wds_dataset(
            wds_path,
            args.batch_size,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=False,
            transform=training_preset((args.size, args.size), args.aug_level, rgb_values),
        )
        (wds_path, _) = fs_ops.wds_braces_from_path(Path(args.val_path))
        if args.wds_val_size is not None:
            dataset_size = args.wds_val_size

        else:
            dataset_size = wds_size(wds_path, device)

        validation_dataset = make_wds_dataset(
            wds_path,
            args.batch_size,
            dataset_size=dataset_size,
            shuffle=False,
            samples_names=False,
            transform=inference_preset((args.size, args.size), rgb_values, 1.0),
        )
        if args.wds_class_file is None:
            args.wds_class_file = str(Path(args.data_path).joinpath(settings.CLASS_LIST_NAME))

        ds_class_to_idx = fs_ops.read_class_file(args.wds_class_file)
        assert class_to_idx == ds_class_to_idx

    else:
        training_dataset = ImageFolder(
            args.data_path,
            transform=training_preset((args.size, args.size), args.aug_level, rgb_values),
            loader=read_image,
        )
        validation_dataset = ImageFolder(
            args.val_path,
            transform=inference_preset((args.size, args.size), rgb_values, 1.0),
            loader=read_image,
            allow_empty=True,
        )
        assert training_dataset.class_to_idx == validation_dataset.class_to_idx
        ds_class_to_idx = training_dataset.class_to_idx
        assert class_to_idx == ds_class_to_idx

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logging.info(f"Using device {device}:{device_id}")
    logging.info(f"Training on {len(training_dataset):,} samples")
    logging.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)
    batch_size: int = args.batch_size
    begin_epoch = 1
    epochs = args.epochs + 1
    if args.stop_epoch is None:
        args.stop_epoch = epochs
    else:
        args.stop_epoch += 1

    # Set data iterators
    if args.mixup_alpha is not None or args.cutmix is True:
        logging.debug("Mixup / cutmix collate activated")
        t = get_mixup_cutmix(args.mixup_alpha, num_outputs, args.cutmix)

        def collate_fn(batch: Any) -> Any:
            return t(*default_collate(batch))

    else:
        collate_fn = None  # type: ignore

    # Initialize student network
    sample_shape = (batch_size,) + (args.channels, args.size, args.size)  # B, C, H, W
    student_name = get_network_name(args.student, net_param=args.student_param, tag=args.student_tag)

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (student, class_to_idx_saved, optimizer_state, scheduler_state, scaler_state) = fs_ops.load_checkpoint(
            device,
            args.student,
            net_param=args.student_param,
            tag=args.student_tag,
            epoch=args.resume_epoch,
            new_size=args.size,
        )
        assert class_to_idx == class_to_idx_saved

    else:
        student = registry.net_factory(
            args.student,
            sample_shape[1],
            num_outputs,
            args.student_param,
            args.size,
        ).to(device)

    if args.fast_matmul is True:
        torch.set_float32_matmul_precision("high")

    # Compile teacher network
    if args.compile is True:
        teacher = torch.compile(teacher)
        student = torch.compile(student)

    # Define loss criteria, optimizer, learning rate scheduler and training parameter groups
    custom_keys_weight_decay = training_utils.get_wd_custom_keys(args)
    parameters = training_utils.optimizer_parameter_groups(
        student,
        args.wd,
        norm_weight_decay=args.norm_wd,
        custom_keys_weight_decay=custom_keys_weight_decay,
        layer_decay=args.layer_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing_alpha)
    if distillation_type == "soft":
        distillation_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    elif distillation_type == "hard":
        distillation_criterion = torch.nn.CrossEntropyLoss()
    elif distillation_type == "deit":
        distillation_criterion = torch.nn.CrossEntropyLoss()
        student.set_distillation_output()
    else:
        raise ValueError(f"Unknown KD type: {args.type}")

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

    elif args.load_scheduler is True:
        scheduler.load_state_dict(scheduler_state)
        last_lr = scheduler.get_last_lr()[0]
        for g in optimizer.param_groups:
            g["lr"] = last_lr

    last_lr = scheduler.get_last_lr()[0]

    # Distributed
    net_without_ddp = student
    if args.distributed is True:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
        net_without_ddp = student.module

    # Model EMA
    if args.model_ema is True:
        model_ema = training_utils.ema_model(args, net_without_ddp, device=device)
        model_to_save = model_ema.module
        eval_model = model_ema

    else:
        model_to_save = net_without_ddp
        eval_model = student

    if args.compile is True and hasattr(model_to_save, "_orig_mod") is True:
        model_to_save = model_to_save._orig_mod  # pylint: disable=protected-access

    # Define metrics
    training_metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy("multiclass", num_classes=num_outputs),
            f"top_{settings.TOP_K}": torchmetrics.Accuracy("multiclass", num_classes=num_outputs, top_k=settings.TOP_K),
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
    training_log_name = training_utils.training_log_name(student_name, device)
    training_log_path = settings.TRAINING_RUNS_PATH.joinpath(training_log_name)
    logging.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_signature(input_shape=sample_shape, num_outputs=num_outputs)
    if args.rank == 0:
        with torch.no_grad():
            summary_writer.add_graph(net_for_info, torch.rand(sample_shape, device=device))

        summary_writer.flush()
        fs_ops.write_signature(student_name, signature)
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

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            shuffle=False,  # Shuffle is done at the wds dataset
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=collate_fn,
            world_size=args.world_size,
            pin_memory=True,
        )

        validation_loader = make_wds_loader(
            validation_dataset,
            batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=collate_fn,
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
    grad_accum_steps: int = args.grad_accum_steps

    # Enable or disable the autograd anomaly detection
    torch.autograd.set_detect_anomaly(args.grad_anomaly_detection)

    # Training loop
    logging.info(f"Starting training with learning rate of {last_lr}")
    for epoch in range(begin_epoch, args.stop_epoch):
        tic = time.time()
        student.train()
        running_loss = 0.0
        running_val_loss = 0.0
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
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                with torch.inference_mode():
                    teacher_outputs = teacher(inputs)

                softmax_teacher = F.softmax(teacher_outputs / args.temperature, dim=-1)
                if distillation_type == "soft":
                    output = student(inputs)
                    dist_output = F.log_softmax(output / args.temperature, dim=-1)
                elif distillation_type == "hard":
                    output = student(inputs)
                    dist_output = output
                elif distillation_type == "deit":
                    (output, dist_output) = torch.unbind(student(inputs), dim=1)
                else:
                    raise RuntimeError

                dist_loss = distillation_criterion(dist_output, softmax_teacher) * (args.temperature**2)
                target_loss = criterion(output, targets)
                loss = (1 - args.lambda_param) * target_loss + (args.lambda_param * dist_loss)

            # if grad_accum_steps > 1:
            #     loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                loss.backward()
                if optimizer_update is True:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

            # Exponential moving average
            if args.model_ema is True and i % args.model_ema_steps == 0:
                model_ema.update_parameters(student)
                if epoch < args.warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)  # pylint: disable=no-member

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)

            training_metrics(output, targets)

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                summary_writer.add_scalars(
                    "loss",
                    {f"training{args.rank}": running_loss / (i * batch_size)},
                    ((epoch - 1) * len(training_dataset)) + (i * batch_size * args.world_size),
                )

                training_metrics_dict = training_metrics.compute()
                if args.rank == 0:
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

        epoch_loss = running_loss / len(training_dataset)

        # Epoch training metrics
        epoch_loss = training_utils.reduce_across_processes(epoch_loss, device)
        logging.info(f"Epoch {epoch}/{epochs-1} training_loss: {epoch_loss:.4f}")

        for metric, value in training_metrics.compute().items():
            logging.info(f"Epoch {epoch}/{epochs-1} {metric}: {value:.4f}")

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
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    outputs = eval_model(inputs)
                    val_loss = criterion(outputs, targets)

                # Statistics
                running_val_loss += val_loss.item() * inputs.size(0)
                validation_metrics(outputs, targets)

                # Update progress bar
                if args.rank == 0:
                    progress.update(n=batch_size * args.world_size)

        if args.rank == 0:
            progress.close()

        epoch_val_loss = running_val_loss / len(validation_dataset)
        epoch_val_loss = training_utils.reduce_across_processes(epoch_val_loss, device)
        validation_metrics_dict = validation_metrics.compute()

        # Learning rate scheduler update
        scheduler.step()
        if last_lr != scheduler.get_last_lr()[0]:
            last_lr = scheduler.get_last_lr()[0]
            logging.info(f"Updated learning rate to: {last_lr}")

        if args.rank == 0:
            summary_writer.add_scalars("loss", {"validation": epoch_val_loss}, epoch * len(training_dataset))
            for metric, value in validation_metrics_dict.items():
                summary_writer.add_scalars("performance", {metric: value}, epoch * len(training_dataset))

            # Epoch validation metrics
            logging.info(f"Epoch {epoch}/{epochs-1} validation_loss (target only): {epoch_val_loss:.4f}")
            for metric, value in validation_metrics_dict.items():
                logging.info(f"Epoch {epoch}/{epochs-1} {metric}: {value:.4f}")

            # Checkpoint model
            if epoch % args.save_frequency == 0:
                fs_ops.checkpoint_model(
                    student_name,
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
            student_name,
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
        description="Train classification model using Knowledge Distillation",
        epilog=(
            "Usage examples:\n"
            "python train_kd.py --type soft --teacher convnext_v2_tiny --teacher-epoch 0 --student regnet "
            "--student-param 1.6 --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 "
            "--size 256 --epochs 100 --wd 0.00005 --mixup-alpha 0.2 --aug-level 3\n"
            "python train_kd.py --type hard --teacher convnext_v2_base --student mobilenet_v4_m --lr 0.8 "
            "--lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 "
            "--wd 0.00005 --mixup-alpha 0.2 --aug-level 3\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("--type", type=str, choices=typing.get_args(DistType), help="type of distillation")
    parser.add_argument("--teacher", type=str, help="the teacher network")
    parser.add_argument("--teacher-param", type=float, help="network specific parameter (teacher)")
    parser.add_argument("--teacher-tag", type=str, help="teacher training log tag (loading only)")
    parser.add_argument("--pts", default=False, action="store_true", help="load torchscript teacher")
    parser.add_argument("--pt2", default=False, action="store_true", help="load pt2 teacher")
    parser.add_argument("--teacher-epoch", type=int, help="load teacher weights from selected epoch")
    parser.add_argument("--student", type=str, help="the student network to train")
    parser.add_argument("--student-param", type=float, help="network specific parameter (student)")
    parser.add_argument("--compile", default=False, action="store_true", help="enable teacher compilation")
    parser.add_argument(
        "--opt",
        type=str,
        choices=list(typing.get_args(training_utils.OptimizerType)),
        default="sgd",
        help="optimizer to use",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="base learning rate")
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
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="number of steps to accumulate gradients")
    parser.add_argument("--channels", type=int, default=3, help="no. of image channels")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="image size (defaults to teacher network size) shared by both networks",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="number of warmup epochs")
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
    parser.add_argument("--aa", default=False, action="store_true", help="Use AutoAugment policy (ignoring aug-level)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=5.0,
        help="controls the smoothness of the output distributions (only used in 'soft')",
    )
    parser.add_argument("--lambda-param", type=float, default=0.5, help="importance of the distillation loss")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
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
    parser.add_argument("--student-tag", type=str, help="add student training logs tag")
    parser.add_argument("--log-interval", type=int, default=50, help="how many steps between summary writes")
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=8,
        help="number of preprocessing workers",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="number of batches loaded in advance by each worker",
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
        "--val-path", type=str, default=str(settings.VALIDATION_DATA_PATH), help="validation directory path"
    )
    parser.add_argument(
        "--data-path", type=str, default=str(settings.TRAINING_DATA_PATH), help="training directory path"
    )
    parser.add_argument("--wds", default=False, action="store_true", help="use webdataset for training")
    parser.add_argument("--wds-class-file", type=str, default=None, help="class list file")
    parser.add_argument("--wds-train-size", type=int, help="size of the wds training set")
    parser.add_argument("--wds-val-size", type=int, help="size of the wds validation set")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.teacher is not None
    assert args.student is not None
    assert 0.5 > args.smoothing_alpha >= 0, "Smoothing alpha must be in range of [0, 0.5)"
    assert args.load_states is False or (
        args.load_states is True and args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert (
        args.load_scheduler is False or args.resume_epoch is not None
    ), "Load scheduler must be from resumed training (--resume-epoch)"
    assert args.wds is False or args.ra_sampler is False, "Repeated Augmentation not currently supported with wds"
    assert (
        registry.exists(args.teacher, task=Task.IMAGE_CLASSIFICATION) is True
    ), "Unknown teacher network, see list-models tool for available options"
    assert (
        registry.exists(args.student, task=Task.IMAGE_CLASSIFICATION) is True
    ), "Unknown student network, see list-models tool for available options"


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
