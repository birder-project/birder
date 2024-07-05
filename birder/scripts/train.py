import argparse
import json
import logging
import sys
import time
import typing
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
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
from birder.common import training_utils
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.core.dataloader.webdataset import make_wds_loader
from birder.core.datasets.webdataset import make_wds_dataset
from birder.core.datasets.webdataset import wds_size
from birder.core.net.base import _REGISTERED_NETWORKS
from birder.core.net.base import get_signature
from birder.core.net.base import net_factory
from birder.core.transforms.classification import RGBMode
from birder.core.transforms.classification import get_mixup_cutmix
from birder.core.transforms.classification import get_rgb_values
from birder.core.transforms.classification import inference_preset
from birder.core.transforms.classification import training_preset


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    torch.backends.cudnn.benchmark = True

    rgb_values = get_rgb_values(args.rgb_mode)
    if args.wds is True:
        (wds_path, _) = cli.wds_braces_from_path(Path(args.data_path))
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
            transform=training_preset(args.size, args.aug_level, rgb_values),
        )
        (wds_path, _) = cli.wds_braces_from_path(Path(args.val_path))
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
            transform=inference_preset(args.size, 1.0, rgb_values),
        )
        if args.wds_class_file is None:
            args.wds_class_file = str(Path(args.data_path).joinpath(settings.CLASS_LIST_NAME))

        class_to_idx = cli.read_class_file(args.wds_class_file)

    else:
        training_dataset = ImageFolder(
            args.data_path,
            transform=training_preset(args.size, args.aug_level, rgb_values),
            loader=read_image,
        )
        validation_dataset = ImageFolder(
            args.val_path,
            transform=inference_preset(args.size, 1.0, rgb_values),
            loader=read_image,
        )
        class_to_idx = training_dataset.class_to_idx

    assert args.model_ema is False or args.model_ema_steps <= len(training_dataset) / args.batch_size

    logging.info(f"Using device {device}:{device_id}")
    logging.info(f"Training on {len(training_dataset):,} samples")
    logging.info(f"Validating on {len(validation_dataset):,} samples")

    num_outputs = len(class_to_idx)
    batch_size: int = args.batch_size
    begin_epoch = 1
    epochs = args.epochs + 1

    # Set data iterators
    if args.mixup_alpha is not None or args.cutmix is True:
        logging.debug("Mixup / cutmix collate activated")
        t = get_mixup_cutmix(args.mixup_alpha, num_outputs, args.cutmix)

        def collate_fn(batch: Any) -> Any:
            return t(*default_collate(batch))

    else:
        collate_fn = None  # type: ignore

    # Initialize network
    sample_shape = (batch_size,) + (3, args.size, args.size)  # B, C, H, W
    network_name = get_network_name(args.network, net_param=args.net_param, tag=args.tag)

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, class_to_idx_saved, optimizer_state, scheduler_state, scaler_state) = cli.load_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            tag=args.tag,
            epoch=args.resume_epoch,
            new_size=args.size,
        )
        if args.reset_head is True:
            net.reset_classifier(len(class_to_idx))
            net.to(device)
            net.freeze(freeze_classifier=False)

        else:
            assert class_to_idx == class_to_idx_saved

    else:
        net = net_factory(args.network, sample_shape[1], num_outputs, args.net_param, args.size).to(device)

    # Compile network
    if args.compile is True:
        net = torch.compile(net)

    # Define loss criteria, optimizer and learning rate scheduler
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))

    if args.transformer_embedding_decay is not None:
        for key in [
            "cls_token",
            "class_token",
            "pos_embedding",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    parameters = training_utils.optimizer_parameter_groups(
        net,
        args.wd,
        norm_weight_decay=args.norm_wd,
        custom_keys_weight_decay=custom_keys_weight_decay,
        layer_decay=args.layer_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing_alpha)
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

    # Gradient scaler
    if args.amp is True:
        scaler = torch.cuda.amp.GradScaler()

    else:
        scaler = None

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
    if args.plot_lr is True:
        logging.info("Fast forwarding scheduler...")
        lrs = []
        for _ in range(begin_epoch, epochs):
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
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally
        # proposed at: https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = training_utils.ExponentialMovingAverage(net_without_ddp, device=device, decay=1.0 - alpha)

        model_to_save = model_ema.module
        eval_model = model_ema

    else:
        model_to_save = net_without_ddp
        eval_model = net

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
    training_log_name = training_utils.training_log_name(network_name, device)
    training_log_path = settings.TRAINING_RUNS_PATH.joinpath(training_log_name)
    logging.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_signature(input_shape=sample_shape, num_outputs=num_outputs)
    if args.rank == 0:
        with torch.no_grad():
            summary_writer.add_graph(net_for_info, torch.rand(sample_shape, device=device))

        summary_writer.flush()
        cli.write_signature(network_name, signature)
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
    if args.distributed is True:
        if args.ra_sampler is True:
            train_sampler = training_utils.RASampler(
                training_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                repetitions=args.ra_reps,
            )

        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)

        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=False)

    else:
        train_sampler = torch.utils.data.RandomSampler(training_dataset)
        validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            shuffle=True,
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

    # Training loop
    logging.info(f"Starting training with learning rate of {last_lr}")
    for epoch in range(begin_epoch, epochs):
        tic = time.time()
        net.train()
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

        for i, (inputs, targets) in enumerate(training_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward and optimize
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                # gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in net.parameters()]), 2)
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                # gn = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in net.parameters()]), 2)
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
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)

            training_metrics(outputs, targets)

            # Write statistics
            if i % 50 == 49:
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
                with torch.cuda.amp.autocast(enabled=args.amp):
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
            logging.info(f"Epoch {epoch}/{epochs-1} validation_loss: {epoch_val_loss:.4f}")
            for metric, value in validation_metrics_dict.items():
                logging.info(f"Epoch {epoch}/{epochs-1} {metric}: {value:.4f}")

            # Checkpoint model
            if epoch % args.save_frequency == 0:
                cli.checkpoint_model(
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
        cli.checkpoint_model(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train classification model",
        epilog=(
            "Usage examples:\n"
            "python train.py --network vgg -p 11 --momentum 0\n"
            "python train.py --network resnet_v2 --net-param 50 --nesterov --lr-scheduler cosine "
            "--size 288 --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3\n"
            "python train.py --network inception_resnet_v2 --nesterov --lr-scheduler cosine "
            "--batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3\n"
            "python train.py --network inception_resnet_v2 --opt adamw --lr 0.0001 --wd 0.01 --epochs 105 "
            "--save-frequency 1 --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 "
            "--resume-epoch 100\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        choices=list(_REGISTERED_NETWORKS.keys()),
        required=True,
        help="the neural network to use",
    )
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required by most networks")
    parser.add_argument(
        "--reset-head",
        default=False,
        action="store_true",
        help="reset classification head and freeze all other layers",
    )
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
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
    parser.add_argument("--size", type=int, default=None, help="image size (defaults to network recommendation)")
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
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="calculated",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
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
        default=0.9998,
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
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.cuda.amp for mixed precision training"
    )
    parser.add_argument("--world-size", type=int, default=1, help="number of distributed processes")
    parser.add_argument("--dist-url", type=str, default="env://", help="url used to set up distributed training")
    parser.add_argument("--clip-grad-norm", type=float, default=None, help="the maximum gradient norm")
    parser.add_argument("--gpu", type=int, help="gpu id to use (ignored in distributed mode)")
    parser.add_argument(
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
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
    args = parser.parse_args()

    assert 0.5 > args.smoothing_alpha >= 0, "Smoothing alpha must be in range of [0, 0.5)"
    assert (
        args.load_states is False or args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert (
        args.load_scheduler is False or args.resume_epoch is not None
    ), "Load scheduler must be from resumed training (--resume-epoch)"
    assert args.wds is False or args.ra_sampler is False, "Repeated Augmentation not currently supported with wds"

    if settings.MODELS_DIR.exists() is False:
        logging.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    training_utils.init_distributed_mode(args)
    if args.size is None:
        args.size = _REGISTERED_NETWORKS[args.network].default_size

    logging.info(f"Using size={args.size}")
    train(args)


if __name__ == "__main__":
    main()