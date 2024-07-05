import argparse
import json
import logging
import sys
import time
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from birder.common import cli
from birder.common import training_utils
from birder.common.lib import get_network_name
from birder.common.lib import get_pretrain_network_name
from birder.conf import settings
from birder.core.dataloader.webdataset import make_wds_loader
from birder.core.datasets.directory import ImageListDataset
from birder.core.datasets.webdataset import make_wds_dataset
from birder.core.datasets.webdataset import wds_size
from birder.core.net.base import PreTrainEncoder
from birder.core.net.base import get_signature
from birder.core.net.base import net_factory
from birder.core.net.base import network_names_filter
from birder.core.net.pretraining.base import _REGISTERED_PRETRAIN_NETWORKS
from birder.core.net.pretraining.base import pretrain_net_factory
from birder.core.transforms.classification import RGBMode
from birder.core.transforms.classification import get_rgb_values
from birder.core.transforms.classification import training_preset

NUM_OUTPUTS = 2  # Just a place holder for easy encoder loading


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    torch.backends.cudnn.benchmark = True

    rgb_values = get_rgb_values(args.rgb_mode)
    if args.wds is True:
        (wds_path, _) = cli.wds_braces_from_path(Path(args.data_path[0]))
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
        input_idx = 0

    else:
        samples = cli.samples_from_paths(args.data_path, class_to_idx={})
        training_dataset = ImageListDataset(samples, transforms=training_preset(args.size, args.aug_level, rgb_values))
        input_idx = 1

    logging.info(f"Using device {device}:{device_id}")
    logging.info(f"Training on {len(training_dataset):,} samples")

    batch_size: int = args.batch_size
    begin_epoch = 1
    epochs = args.epochs + 1

    # Initialize network
    sample_shape = (batch_size,) + (3, args.size, args.size)  # B, C, H, W
    encoder_name = get_network_name(args.encoder, net_param=args.encoder_param, tag="pretrained")
    network_name = get_pretrain_network_name(
        args.network,
        net_param=args.net_param,
        encoder=args.encoder,
        encoder_param=args.encoder_param,
        tag=args.tag,
    )

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, optimizer_state, scheduler_state, scaler_state) = cli.load_pretrain_checkpoint(
            device,
            args.network,
            net_param=args.net_param,
            encoder=args.encoder,
            encoder_param=args.encoder_param,
            tag=args.tag,
            epoch=args.resume_epoch,
        )

    else:
        encoder = net_factory(args.encoder, sample_shape[1], NUM_OUTPUTS, args.encoder_param, args.size)
        net = pretrain_net_factory(args.network, encoder, args.net_param, args.size).to(device)

    # Compile network
    if args.compile is True:
        net = torch.compile(net)

    parameters = training_utils.optimizer_parameter_groups(net, args.wd)
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

    last_lr = scheduler.get_last_lr()[0]
    if args.plot_lr is True:
        logging.info("Fast forwarding scheduler...")
        lrs = []
        for epoch in range(begin_epoch, epochs):
            optimizer.step()
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        plt.plot(range(begin_epoch, epochs), lrs)
        plt.show()
        raise SystemExit(0)

    # Distributed
    net_without_ddp = net
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
        net_without_ddp = net.module

    model_to_save = net_without_ddp
    if args.compile is True and hasattr(model_to_save, "_orig_mod") is True:
        model_to_save = model_to_save._orig_mod  # pylint: disable=protected-access

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

    signature = get_signature(input_shape=sample_shape, num_outputs=NUM_OUTPUTS)
    if args.rank == 0:
        summary_writer.flush()
        cli.write_signature(network_name, signature)
        with open(training_log_path.joinpath("args.json"), "w", encoding="utf-8") as handle:
            json.dump({"cmdline": " ".join(sys.argv), **vars(args)}, handle, indent=2)

        with open(training_log_path.joinpath("training_data.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"training_samples": len(training_dataset)},
                handle,
                indent=2,
            )

    # Data loaders and samplers
    if args.distributed is True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)

    else:
        train_sampler = torch.utils.data.RandomSampler(training_dataset)

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            shuffle=True,
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
            pin_memory=True,
        )

    # Training loop
    logging.info(f"Starting training with learning rate of {last_lr}")
    for epoch in range(begin_epoch, epochs):
        tic = time.time()
        net.train()
        running_loss = 0.0

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

        for i, data in enumerate(training_loader):
            inputs = data[input_idx]
            inputs = inputs.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward and optimize
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs: dict[str, torch.Tensor] = net(inputs)
                loss = outputs["loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)

            # Write statistics
            if i % 20 == 19:
                summary_writer.add_scalars(
                    "loss",
                    {f"training{args.rank}": running_loss / (i * batch_size)},
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

        # Learning rate scheduler update
        scheduler.step()
        if last_lr != scheduler.get_last_lr()[0]:
            last_lr = scheduler.get_last_lr()[0]
            logging.info(f"Updated learning rate to: {last_lr}")

        if args.rank == 0:
            # Checkpoint model
            if epoch % args.save_frequency == 0:
                cli.checkpoint_model(
                    network_name,
                    epoch,
                    model_to_save,
                    signature,
                    {},
                    rgb_values,
                    optimizer,
                    scheduler,
                    scaler,
                )
                cli.checkpoint_model(
                    encoder_name,
                    epoch,
                    model_to_save.encoder,
                    signature,
                    {},
                    rgb_values,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                )

        # Epoch timing
        toc = time.time()
        (minutes, seconds) = divmod(toc - tic, 60)
        logging.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logging.info("---")

    summary_writer.close()

    # Checkpoint model
    if args.distributed is False or (args.distributed is True and args.rank == 0):
        cli.checkpoint_model(
            network_name,
            epoch,
            model_to_save,
            signature,
            {},
            rgb_values,
            optimizer,
            scheduler,
            scaler,
        )
        cli.checkpoint_model(
            encoder_name,
            epoch,
            model_to_save.encoder,
            signature,
            {},
            rgb_values,
            optimizer,
            scheduler,
            scaler,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Pre-train model",
        epilog=(
            "Usage examples:\n"
            "python train_pretrain.py --network mae_vit --encoder vit --encoder-param 2 "
            "--batch-size 32 --opt adamw --lr 0.0001\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        choices=list(_REGISTERED_PRETRAIN_NETWORKS.keys()),
        required=True,
        help="the neural network to use",
    )
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required by most networks")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=network_names_filter(PreTrainEncoder),
        required=True,
        help="the neural network to used as encoder (network being pre-trained)",
    )
    parser.add_argument(
        "--encoder-param",
        type=float,
        help="network specific parameter, required by most networks (for the encoder)",
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
    parser.add_argument("--batch-size", type=int, default=32, help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="number of warmup epochs")
    parser.add_argument(
        "--aug-level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=1,
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
    parser.add_argument("--save-frequency", type=int, default=2, help="frequency of model saving")
    parser.add_argument("--resume-epoch", type=int, help="epoch to resume training from")
    parser.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
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
    parser.add_argument("--data-path", nargs="+", help="training directories paths (directories and files)")
    parser.add_argument("--wds", default=False, action="store_true", help="use webdataset for training")
    parser.add_argument("--wds-train-size", type=int, help="size of the wds training set")
    args = parser.parse_args()

    assert args.load_states is False or (
        args.load_states is True and args.resume_epoch is not None
    ), "Load states must be from resumed training (--resume-epoch)"
    assert args.wds is False or len(args.data_path) == 1, "WDS must be a single directory"

    if settings.MODELS_DIR.exists() is False:
        logging.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    training_utils.init_distributed_mode(args)
    if args.size is None:
        # Prefer pretrain size over encoder default size
        args.size = _REGISTERED_PRETRAIN_NETWORKS[args.network].default_size

    logging.info(f"Using size={args.size}")
    train(args)


if __name__ == "__main__":
    main()