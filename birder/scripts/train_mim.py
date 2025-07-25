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
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.folder import pil_loader  # Slower but Handles external dataset quirks better
from tqdm import tqdm

from birder.common import cli
from birder.common import fs_ops
from birder.common import training_cli
from birder.common import training_utils
from birder.common.lib import format_duration
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.common.lib import set_random_seeds
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.directory import tv_loader
from birder.data.datasets.fake import FakeDataWithPaths
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import get_rgb_stats
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import PreTrainEncoder
from birder.net.base import get_signature
from birder.net.mim.base import get_mim_signature

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    #
    # Initialize
    #
    training_utils.init_distributed_mode(args)
    if args.size is None:
        # Prefer mim size over encoder default size
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
    else:
        disable_tqdm = None  # Let tqdm auto-detect (None = auto)

    # Enable or disable the autograd anomaly detection
    torch.autograd.set_detect_anomaly(args.grad_anomaly_detection)

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    logger.debug(f"Using RGB stats: {rgb_stats}")

    training_transform = training_utils.get_training_transform(args)
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

    batch_size: int = args.batch_size
    logger.debug(f"Effective batch size = {args.batch_size * args.grad_accum_steps * args.world_size}")

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
    encoder_name = get_network_name(args.encoder, net_param=args.encoder_param, tag="mim")
    network_name = get_mim_network_name(
        args.network,
        net_param=args.net_param,
        encoder=args.encoder,
        encoder_param=args.encoder_param,
        tag=args.tag,
    )

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, training_states) = fs_ops.load_mim_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            config=args.model_config,
            encoder=args.encoder,
            encoder_param=args.encoder_param,
            encoder_config=args.encoder_model_config,
            tag=args.tag,
            epoch=args.resume_epoch,
        )

    elif args.pretrained is True:
        fs_ops.download_model_by_weights(network_name, progress_bar=training_utils.is_local_primary(args))
        (net, training_states) = fs_ops.load_mim_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            config=args.model_config,
            encoder=args.encoder,
            encoder_param=args.encoder_param,
            encoder_config=args.encoder_model_config,
            tag=args.tag,
            epoch=None,
        )

    else:
        encoder = registry.net_factory(
            args.encoder,
            sample_shape[1],
            0,
            net_param=args.encoder_param,
            config=args.encoder_model_config,
            size=args.size,
        )
        net = registry.mim_net_factory(
            args.network, encoder, net_param=args.net_param, config=args.model_config, size=args.size
        )
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
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
    net_without_ddp = net
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters
        )
        net_without_ddp = net.module

    model_to_save = net_without_ddp
    if args.compile is True and hasattr(model_to_save, "_orig_mod") is True:
        model_to_save = model_to_save._orig_mod  # pylint: disable=protected-access

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
    training_log_name = training_utils.training_log_name(network_name, device)
    training_log_path = settings.TRAINING_RUNS_PATH.joinpath(training_log_name)
    logger.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_mim_signature(input_shape=sample_shape)
    encoder_signature = get_signature(input_shape=sample_shape, num_outputs=0)
    file_handler: logging.Handler = logging.NullHandler()
    if training_utils.is_local_primary(args) is True:
        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        file_handler = training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
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
        for i, (_, inputs, _) in enumerate(training_loader):
            inputs = inputs.to(device, dtype=model_dtype, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                outputs: dict[str, torch.Tensor] = net(inputs)
                loss = outputs["loss"]

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

            # Statistics
            running_loss.update(loss.detach())

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                time_now = time.time()
                time_cost = time_now - start_time
                rate = (i - last_idx) * (batch_size * args.world_size) / time_cost
                start_time = time_now
                last_idx = i
                cur_lr = max(scheduler.get_last_lr())

                running_loss.synchronize_between_processes(device)
                with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                    log.info(
                        f"[Training] Epoch {epoch}/{epochs-1}, step {i+1}/{last_batch_idx}  "
                        f"Loss: {running_loss.avg:.4f}  "
                        f"Elapsed: {format_duration(time_now-epoch_start)}  "
                        f"Time: {time_cost:.1f}s  "
                        f"Rate: {rate:.1f} samples/s  "
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
        epoch_loss = running_loss.global_avg
        logger.info(f"[Training] Epoch {epoch}/{epochs-1} training_loss: {epoch_loss:.4f}")

        # Learning rate scheduler update
        if iter_update is False:
            scheduler.step()
        if last_lr != max(scheduler.get_last_lr()):
            last_lr = max(scheduler.get_last_lr())
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
                    encoder_name,
                    epoch,
                    model_to_save.encoder,
                    encoder_signature,
                    {},
                    rgb_stats,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    model_base=None,
                )
                if args.keep_last is not None:
                    fs_ops.clean_checkpoints(network_name, args.keep_last)
                    fs_ops.clean_checkpoints(encoder_name, args.keep_last)

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
            encoder_name,
            epoch,
            model_to_save.encoder,
            encoder_signature,
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
            "torchrun --nproc_per_node=2 train_mim.py \\\n"
            "    --network mae_vit \\\n"
            "    --encoder vit_b16 \\\n"
            "    --opt adamw \\\n"
            "    --lr 0.00015 \\\n"
            "    --opt-betas 0.9 0.95 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --warmup-epochs 40 \\\n"
            "    --batch-size 256 \\\n"
            "    --wd 0.05 \\\n"
            "    --encoder-model-config drop_path_rate=0.0 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
            "    --compile-opt \\\n"
            "    --data-path data/training data/raw_data ~/Datasets\n"
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
    parser.add_argument("-t", "--tag", type=str, help="add model tag")
    parser.add_argument("--encoder", type=str, help="the neural network to used as encoder (network being pre-trained)")
    parser.add_argument(
        "--encoder-param",
        type=float,
        help="network specific parameter, required by some networks (for the encoder)",
    )
    parser.add_argument(
        "--encoder-model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the encoder default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    training_cli.add_optimization_args(parser)
    training_cli.add_lr_wd_args(parser)
    training_cli.add_lr_scheduler_args(parser)
    training_cli.add_training_schedule_args(parser, default_epochs=800)
    training_cli.add_input_args(parser)
    training_cli.add_data_aug_args(parser, default_level=1, default_min_scale=0.3)
    training_cli.add_dataloader_args(parser, default_drop_last=True)
    training_cli.add_precision_args(parser)
    training_cli.add_compile_args(parser)
    training_cli.add_checkpoint_args(parser, pretrained=True)
    training_cli.add_distributed_args(parser)
    training_cli.add_logging_and_debug_args(parser, default_log_interval=100)
    training_cli.add_training_data_args(parser, unsupervised=True)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_path = [str(p) for p in args.data_path]
    args.size = cli.parse_size(args.size)

    # This will capture the common argument mistakes
    training_cli.common_args_validation(args)

    # Script specific checks
    if registry.exists(args.network, task=Task.MASKED_IMAGE_MODELING) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")
    if registry.exists(args.encoder, net_type=PreTrainEncoder) is False:
        raise cli.ValidationError(f"--encoder {args.network} not supported, see list-models tool for available options")


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
    logger = logging.getLogger(__spec__.name)
    main()
