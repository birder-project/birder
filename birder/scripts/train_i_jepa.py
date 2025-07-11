"""
I-JEPA (Image-based Joint-Embedding Predictive Architecture), adapted from
https://github.com/facebookresearch/ijepa/blob/main/src/train.py

Paper "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
https://arxiv.org/abs/2301.08243

Changes from original:
* Per epoch weight decay scheduling (instead of per step)
"""

# Reference license: Attribution-NonCommercial 4.0 International

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
from birder.net.ssl.i_jepa import I_JEPA
from birder.net.ssl.i_jepa import MultiBlockMasking
from birder.net.ssl.i_jepa import VisionTransformerPredictor
from birder.net.ssl.i_jepa import apply_masks
from birder.net.ssl.i_jepa import repeat_interleave_batch

logger = logging.getLogger(__name__)


class TrainCollator:
    def __init__(
        self,
        mask_generator: Callable[[int], tuple[list[torch.Tensor], list[torch.Tensor]]],
    ) -> None:
        self.mask_generator = mask_generator

    def __call__(self, batch: Any) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        (enc_masks, pred_masks) = self.mask_generator(B)

        return (collated_batch, enc_masks, pred_masks)


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
    encoder_name = get_network_name(args.network, net_param=args.net_param, tag="i-jepa")
    network_name = get_mim_network_name(
        "i_jepa",
        net_param=None,
        encoder=args.network,
        encoder_param=args.net_param,
        tag=args.tag,
    )

    if args.model_config is not None:
        model_config = args.model_config.copy()
        model_config.update({"drop_path_rate": 0.0})
    else:
        model_config = {"drop_path_rate": 0.0}

    backbone = registry.net_factory(
        args.network,
        sample_shape[1],
        0,
        net_param=args.net_param,
        config=model_config,
        size=args.size,
    )
    num_special_tokens = backbone.num_special_tokens
    target_backbone = registry.net_factory(
        args.network,
        sample_shape[1],
        0,
        net_param=args.net_param,
        config=model_config,
        size=args.size,
    )
    encoder = I_JEPA(backbone)
    target_encoder = I_JEPA(target_backbone)
    target_encoder.load_state_dict(encoder.state_dict())

    mask_size = (args.size[0] // encoder.backbone.max_stride, args.size[1] // encoder.backbone.max_stride)
    predictor = VisionTransformerPredictor(
        mask_size,
        encoder.backbone.embedding_size,
        predictor_embed_dim=args.predictor_embed_dim,
        mlp_dim=4 * args.predictor_embed_dim,
        num_heads=args.predictor_num_heads,
        depth=args.predictor_depth,
        drop_path_rate=0.0,
    )

    net = torch.nn.ModuleDict(
        {
            "encoder": encoder,
            "target_encoder": target_encoder,
            "predictor": predictor,
        }
    )
    net.task = encoder.task

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        (net, training_states) = fs_ops.load_simple_checkpoint(device, net, network_name, epoch=args.resume_epoch)
        encoder = net["encoder"]
        target_encoder = net["target_encoder"]
        predictor = net["predictor"]

    else:
        training_states = fs_ops.TrainingStates.empty()

    assert isinstance(encoder.backbone, MaskedTokenOmissionMixin)
    assert isinstance(net, torch.nn.Module)

    net.to(device, dtype=model_dtype)
    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile network
    if args.compile is True:
        # encoder = torch.compile(encoder)  # Dynamic sequence length not handled well by dynamo
        target_encoder = torch.compile(target_encoder)
        # predictor = torch.compile(predictor)

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode)
    mask_generator = MultiBlockMasking(
        mask_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        n_enc=1,
        n_pred=4,
        min_keep=10,  # math.ceil(mask_size[0] * mask_size[1] / 25.6),
        allow_overlap=False,
    )
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

    # Momentum and weight decay schedule
    momentum_schedule = training_utils.cosine_scheduler(0.996, 1.0, args.epochs, 0, last_batch_idx + 1)
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

    # There is no backpropagation through the teacher
    for p in target_encoder.parameters():
        p.requires_grad = False

    encoder_without_ddp = encoder
    # predictor_without_ddp = predictor
    if args.distributed is True:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters
        )
        predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[args.local_rank])
        encoder_without_ddp = encoder.module
        # predictor_without_ddp = predictor.module

    model_to_save = net
    if args.compile is True and hasattr(model_to_save["target_encoder"], "_orig_mod") is True:
        model_to_save["target_encoder"] = model_to_save["target_encoder"]._orig_mod  # pylint: disable=protected-access

    #
    # Misc
    #

    # Print network summary
    net_for_info = encoder_without_ddp.backbone
    if args.compile is True and hasattr(encoder_without_ddp, "_orig_mod") is True:
        net_for_info = encoder_without_ddp._orig_mod.backbone  # pylint: disable=protected-access

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

        if args.distributed is True:
            train_sampler.set_epoch(epoch)

        if wd_schedule is not None:
            wd = wd_schedule[epoch - 1]
            for param_group in optimizer.param_groups:
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd

            logger.info(f"Updated wd to: {wd}")

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

        for i, ((_, images, _), enc_masks, pred_masks) in enumerate(training_loader):
            global_step = ((epoch - 1) * (last_batch_idx + 1)) + i
            images = images.to(device, dtype=model_dtype, non_blocking=True)
            enc_masks = [m.to(device, non_blocking=True) for m in enc_masks]
            pred_masks = [m.to(device, non_blocking=True) for m in pred_masks]

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)

            # Forward, backward and optimize
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                # Target encoder
                with torch.no_grad():
                    h = target_encoder(images)
                    h = h[:, num_special_tokens:, :]  # Remove special tokens
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, pred_masks)
                    h = repeat_interleave_batch(h, batch_size, repeat=len(enc_masks))

                # Context encoder
                # NOTE: When enc_masks > 1, this implementation is not as efficient as the original, as we run through
                # the stem over and over with the same input.
                z = torch.concat([encoder(images, enc_mask) for enc_mask in enc_masks], dim=0)
                z = z[:, num_special_tokens:, :]  # Remove special tokens
                z = predictor(z, enc_masks, pred_masks)

                loss = F.smooth_l1_loss(z, h)

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

            # EMA update for the target encoder
            with torch.no_grad():
                m = momentum_schedule[global_step]
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Statistics
            running_loss.update(loss.detach())

            # Write statistics
            if (i == last_batch_idx) or (i + 1) % args.log_interval == 0:
                running_loss.synchronize_between_processes(device)
                if training_utils.is_local_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
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
                    model_to_save["encoder"].backbone,
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
                    fs_ops.clean_checkpoints(encoder_name, args.keep_last)

        # Epoch timing
        toc = time.time()
        (minutes, seconds) = divmod(toc - tic, 60)
        logger.info(f"Time cost: {int(minutes):0>2}m{seconds:04.1f}s")
        logger.info("---")

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
            model_to_save["encoder"].backbone,
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
            "torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa \\\n"
            "    --network vit_reg4_b16 \\\n"
            "    --opt adamw \\\n"
            "    --lr 0.001 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --lr-cosine-min 1e-6 \\\n"
            "    --warmup-epochs 40 \\\n"
            "    --batch-size 128 \\\n"
            "    --wd 0.04 \\\n"
            "    --wd-end 0.4 \\\n"
            "    --norm-wd 0 \\\n"
            "    --bias-weight-decay 0 \\\n"
            "    --amp \\\n"
            "    --compile \\\n"
            "    --compile-opt \\\n"
            "    --data-path data/training data/raw_data\n"
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
    parser.add_argument("--predictor-embed-dim", type=int, default=384, help="predictor embedding dimension")
    parser.add_argument("--predictor-num-heads", type=int, default=12, help="predictor number of heads")
    parser.add_argument("--predictor-depth", type=int, default=12, help="predictor number of layers")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-opt", default=False, action="store_true", help="enable compilation for optimizer step"
    )
    training_utils.add_optimizer_args(parser)
    training_utils.add_lr_wd_args(parser, wd_end=True)
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
    training_utils.add_aug_args(parser, default_level=1, default_min_scale=0.35)
    parser.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="rgb mean and std to use for normalization",
    )
    parser.add_argument("--epochs", type=int, default=300, metavar="N", help="number of training epochs")
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
    parser.add_argument("--load-scheduler", default=False, action="store_true", help="load scheduler only resuming")
    parser.add_argument("-t", "--tag", type=str, help="add training logs tag")
    parser.add_argument(
        "--log-interval", type=int, default=100, metavar="N", help="how many steps between summary writes"
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
