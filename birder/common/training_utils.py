import argparse
import contextlib
import logging
import math
import os
import re
import subprocess
from collections import deque
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional
from typing import Sized

import numpy as np
import torch
import torch.amp
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision.ops import FrozenBatchNorm2d

from birder.conf import settings
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import training_preset
from birder.optim import Lamb
from birder.optim import Lars
from birder.scheduler import CooldownLR

logger = logging.getLogger(__name__)

OptimizerType = Literal["sgd", "rmsprop", "adam", "adamw", "nadam", "nadamw", "lamb", "lambw", "lars"]
SchedulerType = Literal["constant", "step", "multistep", "cosine", "polynomial"]

###############################################################################
# Data Sampling
###############################################################################


class RASampler(torch.utils.data.Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the TorchVision repository:
    https://github.com/pytorch/vision/blob/v0.16.2/references/classification/sampler.py

    Based on: https://arxiv.org/pdf/2105.13343.pdf
    """

    def __init__(
        self,
        dataset: Sized,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int = 0,
        repetitions: int = 3,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle is True:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


###############################################################################
# Model Weight Averaging
###############################################################################


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """
    Maintains moving averages of model parameters using an exponential decay

    See: "Averaging Weights Leads to Wider Optima and Better Generalization"
    https://arxiv.org/abs/1803.05407
    """

    def __init__(self, model: torch.nn.Module, decay: float, device: torch.device) -> None:
        def ema_avg(
            avg_model_param: torch.nn.Parameter, model_param: torch.nn.Parameter, _num_averaged: int
        ) -> torch.Tensor:
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def ema_model(args: argparse.Namespace, net: torch.nn.Module, device: torch.device) -> ExponentialMovingAverage:
    # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally
    # proposed at: https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
    #
    # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
    # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
    # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs

    adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(net, device=device, decay=1.0 - alpha)

    return model_ema


###############################################################################
# Optimizer Parameter Groups
###############################################################################


def group_by_regex(strings: list[str], pattern: str) -> list[list[str]]:
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_block: Optional[tuple[str, ...]] = None

    for s in strings:
        match = re.search(pattern, s)
        if match is not None:
            block_key = match.groups()
            if block_key != current_block:
                if len(current_group) > 0:
                    groups.append(current_group)

                current_group = []
                current_block = block_key

        elif current_block is not None:
            if len(current_group) > 0:
                groups.append(current_group)

            current_group = []
            current_block = None

        current_group.append(s)

    if len(current_group) > 0:
        groups.append(current_group)

    return groups


def count_layers(model: torch.nn.Module) -> int:
    num_layers = 0
    module_stack = [model]
    visited_modules: list[int] = []
    while len(module_stack) > 0:
        skip_module = False
        module = module_stack.pop()
        if id(module) in visited_modules:
            skip_module = True

        visited_modules.append(id(module))
        parameters_found = False
        for _, _ in module.named_parameters(recurse=False):
            if skip_module is True:
                break

            parameters_found = True

        if parameters_found is True:
            num_layers += 1

        for _, child_module in reversed(list(module.named_children())):
            module_stack.append(child_module)

    return num_layers


# pylint: disable=protected-access,too-many-locals,too-many-branches
def optimizer_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    custom_keys_weight_decay: Optional[list[tuple[str, float]]] = None,
    layer_decay: Optional[float] = None,
    layer_decay_min_scale: Optional[float] = None,
    layer_decay_no_opt_scale: Optional[float] = None,
    bias_lr: Optional[float] = None,
    backbone_lr: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Return parameter groups for optimizers with per-parameter group weight decay.

    This function creates parameter groups with customizable weight decay, layer-wise
    learning rate scaling, and special handling for different parameter types. It supports
    advanced optimization techniques like layer decay and custom weight decay rules.

    Referenced from https://github.com/pytorch/vision/blob/main/references/classification/utils.py and from
    https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

    Layer decay optimization trick from: https://github.com/huggingface/pytorch-image-models/pull/2537/files

    Parameters
    ----------
    model
        The PyTorch model whose parameters will be grouped for optimization.
    weight_decay
        Default weight decay (L2 regularization) value applied to parameters.
    norm_weight_decay
        Weight decay value specifically for normalization layers. If None, uses weight_decay.
    custom_keys_weight_decay
        List of (parameter_name, weight_decay) tuples for applying custom weight decay
        values to specific parameters by name matching.
    layer_decay
        Layer-wise learning rate decay factor.
    layer_decay_min_scale
        Minimum learning rate scale factor when using layer decay. Prevents layers from having too small learning rates.
    layer_decay_no_opt_scale
        Learning rate scale threshold below which parameters are frozen (requires_grad=False).
    bias_lr
        Custom learning rate for bias parameters (parameters ending with '.bias').
    backbone_lr
        Custom learning rate for backbone parameters (parameters starting with 'backbone.').

    Returns
    -------
    List of parameter group dictionaries suitable for PyTorch optimizers.

    Notes
    -----
    - The function handles duplicate parameters (module aliases) by warning and skipping them
    - Layer grouping is determined by the model's `block_group_regex` attribute if available,
      otherwise layers are counted sequentially
    - Parameters with requires_grad=False are automatically skipped
    - Custom key matching supports both full parameter names and shortened names for nested modules
    """

    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.RMSNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    block_group_regex = getattr(model, "block_group_regex", None)
    if block_group_regex is not None:
        names = [n for n, _ in model.named_parameters()]
        groups = group_by_regex(names, block_group_regex)
        group_map = {item: index for index, sublist in enumerate(groups) for item in sublist}
        num_layers = len(groups)
    else:
        group_map = {}
        num_layers = count_layers(model)
        if layer_decay is not None:
            logger.warning("Assigning lr scaling (layer decay) without a block group map")

    # Build layer scale
    if layer_decay_min_scale is None:
        layer_decay_min_scale = 0.0

    layer_scales = []
    if layer_decay is not None:
        layer_max = num_layers - 1
        layer_scales = [max(layer_decay_min_scale, layer_decay ** (layer_max - i)) for i in range(num_layers)]
        logger.info(f"Layer scaling in range of {min(layer_scales)} - {max(layer_scales)} on {num_layers} layers")

    # Set weight decay and layer decay
    idx = 0
    params = []
    module_stack_with_prefix = [(model, "")]
    visited_modules = []
    while len(module_stack_with_prefix) > 0:
        skip_module = False
        (module, prefix) = module_stack_with_prefix.pop()
        if id(module) in visited_modules:
            skip_module = True

        visited_modules.append(id(module))
        parameters_found = False
        for name, p in module.named_parameters(recurse=False):
            target_name = f"{prefix}.{name}" if prefix != "" else name
            idx = group_map.get(target_name, idx)
            if skip_module is True:
                break

            parameters_found = True
            if p.requires_grad is False:
                continue
            if layer_decay is not None and layer_decay_no_opt_scale is not None:
                if layer_scales[idx] < layer_decay_no_opt_scale:
                    p.requires_grad_(False)

            is_custom_key = False
            if custom_keys_weight_decay is not None:
                for key, custom_wd in custom_keys_weight_decay:
                    target_name_for_custom_key = f"{prefix}.{name}" if prefix != "" and "." in key else name
                    if key == target_name_for_custom_key:
                        d = {
                            "params": p,
                            "weight_decay": custom_wd,
                            "lr_scale": 1.0 if layer_decay is None else layer_scales[idx],
                        }
                        if backbone_lr is not None and target_name.startswith("backbone.") is True:
                            d["lr"] = backbone_lr

                        params.append(d)
                        is_custom_key = True
                        break

            if is_custom_key is False:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    wd = norm_weight_decay
                else:
                    wd = weight_decay

                d = {
                    "params": p,
                    "weight_decay": wd,
                    "lr_scale": 1.0 if layer_decay is None else layer_scales[idx],
                }
                if backbone_lr is not None and target_name.startswith("backbone.") is True:
                    d["lr"] = backbone_lr

                if bias_lr is not None and target_name.endswith(".bias") is True:
                    d["lr"] = bias_lr

                params.append(d)

        if parameters_found is True:
            idx += 1

        for child_name, child_module in reversed(list(module.named_children())):
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            module_stack_with_prefix.append((child_module, child_prefix))

    return params


def get_wd_custom_keys(args: argparse.Namespace) -> list[tuple[str, float]]:
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))

    if args.transformer_embedding_decay is not None:
        for key in [
            "cls_token",
            "class_token",
            "mask_token",
            "pos_embed",
            "pos_embedding",
            "pos_embed_win",
            "position_embedding",
            "relative_position_bias_table",
            "rel_pos_h",
            "rel_pos_w",
            "decoder_embed",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    return custom_keys_weight_decay


###############################################################################
# Components Setup
###############################################################################


def get_optimizer(parameters: list[dict[str, Any]], l_rate: float, args: argparse.Namespace) -> torch.optim.Optimizer:
    opt: OptimizerType = args.opt
    kwargs = {}
    if getattr(args, "opt_eps", None) is not None:
        kwargs["eps"] = args.opt_eps
    if getattr(args, "opt_betas", None) is not None:
        kwargs["betas"] = args.opt_betas
    if getattr(args, "opt_alpha", None) is not None:
        kwargs["alpha"] = args.opt_alpha

    # For optimizer compilation
    # lr = torch.tensor(l_rate) - Causes weird LR scheduling bugs
    lr = l_rate
    if getattr(args, "compile_opt", False) is not False:
        if opt not in ("lamb", "lambw", "lars"):
            logger.debug("Setting optimizer capturable to True")
            kwargs["capturable"] = True

    if opt == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd
        )
    elif opt == "rmsprop":
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.9
        if "eps" not in kwargs:
            kwargs["eps"] = 0.0316

        optimizer = torch.optim.RMSprop(parameters, lr=lr, momentum=args.momentum, weight_decay=args.wd, **kwargs)
    elif opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=args.wd, **kwargs)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=args.wd, **kwargs)
    elif opt == "nadam":
        optimizer = torch.optim.NAdam(parameters, lr=lr, weight_decay=args.wd, **kwargs)
    elif opt == "nadamw":
        optimizer = torch.optim.NAdam(parameters, lr=lr, weight_decay=args.wd, decoupled_weight_decay=True, **kwargs)
    elif opt == "lamb":
        optimizer = Lamb(parameters, lr=lr, weight_decay=args.wd, **kwargs)
    elif opt == "lambw":
        optimizer = Lamb(parameters, lr=lr, weight_decay=args.wd, decoupled_decay=True, **kwargs)
    elif opt == "lars":
        optimizer = Lars(
            parameters, lr=lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd, **kwargs
        )
    else:
        raise ValueError("Unknown optimizer")

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer, iters_per_epoch: int, args: argparse.Namespace
) -> torch.optim.lr_scheduler.LRScheduler:
    # At first, we translate everything into "steps"
    begin_step = 0
    if args.resume_epoch is not None:
        begin_step = args.resume_epoch * iters_per_epoch

    steps = args.epochs * iters_per_epoch

    if args.warmup_epochs is not None:
        warmup_steps = args.warmup_epochs * iters_per_epoch
    elif args.warmup_iters is not None:
        warmup_steps = args.warmup_iters
    else:
        warmup_steps = 0

    if args.cooldown_epochs is not None:
        cooldown_steps = args.cooldown_epochs * iters_per_epoch
    elif args.cooldown_iters is not None:
        cooldown_steps = args.cooldown_iters
    else:
        cooldown_steps = 0

    # Warmup steps is given in absolute number from 0
    remaining_warmup = max(0, warmup_steps - begin_step)

    # Cooldown steps is given in absolute number from the end
    remaining_cooldown = min(cooldown_steps, steps - begin_step)

    main_steps = steps - begin_step - remaining_warmup - remaining_cooldown - 1

    logger.debug(f"Using {iters_per_epoch} steps per epoch")
    logger.debug(
        f"Scheduler {args.lr_scheduler} set for {steps} steps of which {warmup_steps} "
        f"are warmup and {cooldown_steps} cooldown"
    )
    logger.debug(
        f"Currently starting from step {begin_step} with {remaining_warmup} remaining warmup steps "
        f"and {remaining_cooldown} remaining cooldown steps"
    )
    logger.debug(f"Main scheduler has {main_steps} steps")

    if args.lr_scheduler == "constant":
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    elif args.lr_scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_step_gamma
        )
    elif args.lr_scheduler == "multistep":
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_step_gamma
        )
    elif args.lr_scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_steps, eta_min=args.lr_cosine_min
        )
    elif args.lr_scheduler == "polynomial":
        main_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=main_steps, power=args.lr_power)
    else:
        raise ValueError("Unknown learning rate scheduler")

    # Handle warmup and cooldown
    if warmup_steps > 0 or cooldown_steps > 0:
        schedulers = []
        milestones = []
        if warmup_steps > 0:
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay if remaining_warmup > 0 else 1.0,
                total_iters=remaining_warmup,
            )
            schedulers.append(warmup_lr_scheduler)
            milestones.append(remaining_warmup)

        schedulers.append(main_scheduler)

        if cooldown_steps > 0:
            cooldown_lr_scheduler = CooldownLR(optimizer, total_steps=remaining_cooldown)
            schedulers.append(cooldown_lr_scheduler)
            milestones.append(remaining_warmup + main_steps + 1)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )

    else:
        scheduler = main_scheduler

    return scheduler


def get_amp_scaler(amp: bool, amp_dtype_str: str) -> tuple[Optional[torch.amp.GradScaler], Optional[torch.dtype]]:
    if amp is True:
        scaler = torch.amp.GradScaler("cuda")
        amp_dtype = getattr(torch, amp_dtype_str)

    else:
        scaler = None
        amp_dtype = None

    return (scaler, amp_dtype)


def get_samplers(
    args: argparse.Namespace, training_dataset: torch.utils.data.Dataset, validation_dataset: torch.utils.data.Dataset
) -> torch.utils.data.Sampler:
    if args.distributed is True:
        if args.ra_sampler is True:
            train_sampler = RASampler(
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

    return (train_sampler, validation_sampler)


def get_training_transform(args: argparse.Namespace) -> Callable[..., torch.Tensor]:
    return training_preset(
        args.size,
        args.aug_type,
        args.aug_level,
        get_rgb_stats(args.rgb_mode),
        args.resize_min_scale,
        args.re_prob,
        args.use_grayscale,
        args.ra_num_ops,
        args.ra_magnitude,
        args.augmix_severity,
        args.simple_crop,
    )


###############################################################################
# Metrics Tracking and Reporting
###############################################################################


def to_tensor(x: torch.Tensor | float, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x

    return torch.tensor(x, device=device)


class SmoothedValue:
    # Adapted from: https://github.com/facebookresearch/capi/blob/main/utils.py

    def __init__(self, window_size: int = 32) -> None:
        self.window_size = window_size
        self.deque: deque[torch.Tensor | float] = deque(maxlen=window_size)
        self.total: torch.Tensor | float = 0.0
        self.count: int = 0

    def update(self, value: torch.Tensor | float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device: torch.device, op: dist.ReduceOp = dist.ReduceOp.AVG) -> None:
        if is_dist_available_and_initialized() is False:
            return

        count = to_tensor(self.count).to(device=device, dtype=torch.float64).reshape(1)
        total = to_tensor(self.total).to(device=device, dtype=torch.float64).reshape(1)
        tensor_deque = torch.tensor(list(self.deque), dtype=torch.float64, device=device)
        t = torch.concat([count, total, tensor_deque], dim=0)
        dist.barrier()
        dist.all_reduce(t, op=op)
        self.count = int(t[0].cpu().item())
        self.total = t[1]
        self.deque = deque(list(t[2:]), maxlen=self.window_size)

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().cpu().item()  # type: ignore[no-any-return]

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().cpu().item()  # type: ignore[no-any-return]

    @property
    def global_avg(self) -> float:
        return to_tensor(self.total, torch.device("cpu")).item() / self.count  # type: ignore[no-any-return]

    @property
    def max(self) -> float:
        return torch.tensor(self.deque).max().cpu().item()  # type: ignore[no-any-return]

    @property
    def value(self) -> float:
        v = self.deque[-1]
        return to_tensor(v, torch.device("cpu")).item()  # type: ignore[no-any-return]


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_pred.dim() > 1 and y_pred.size(1) > 1:
        y_pred = y_pred.argmax(dim=1)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return (y_true == y_pred).float().mean().item()  # type: ignore[no-any-return]


###############################################################################
# Distributed Training Utilities
###############################################################################


def init_distributed_mode(args: argparse.Namespace) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torch.distributed.run, torchrun
        logger.debug("Detected PyTorch distributed environment (torchrun / torch.distributed.launch)")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # Slurm
        logger.debug("Detected Slurm distributed environment")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])
        args.local_rank = int(os.environ["SLURM_LOCALID"])
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        # Open MPI
        logger.debug("Detected Open MPI distributed environment")
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif args.world_size > 1:
        # Other, by CLI
        logger.debug("Using distributed mode with CLI-provided parameters")
    else:
        logger.debug("No distributed environment detected, running in single-process mode")
        args.rank = 0
        args.distributed = False
        if args.local_rank is None:
            args.local_rank = 0

        torch.cuda.set_device(args.local_rank)
        return

    torch.cuda.set_device(args.local_rank)

    args.distributed = args.world_size > 1
    if args.distributed is False:
        logger.info("World size is 1, not using distributed mode")
        return

    # Initialize process group
    logger.debug(f"Using {args.dist_backend} backend for distributed training")
    logger.info(f"Distributed init (rank {args.rank}), total {args.world_size}: {args.dist_url}")
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        device_id=torch.device(f"cuda:{args.local_rank}"),
    )

    # Synchronize all processes
    logger.debug("Synchronizing all processes...")
    dist.barrier()
    if is_local_primary(args) is False:
        logger.debug(f"Non-primary process (rank {args.rank}), disabling logging")
        disable_print()
        logging.disable(logging.WARNING)


def shutdown_distributed_mode(args: argparse.Namespace) -> None:
    if args.distributed is True:
        dist.destroy_process_group()


def disable_print() -> None:
    import builtins as __builtin__  # pylint: disable=import-outside-toplevel

    builtin_print = __builtin__.print

    def print(*args, **kwargs):  # type: ignore  # pylint: disable=redefined-builtin
        force = kwargs.pop("force", False)
        if force is True:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized() -> bool:
    if dist.is_available() is False:
        return False
    if dist.is_initialized() is False:
        return False

    return True


def reduce_across_processes(value: torch.Tensor | float, device: torch.device, op: dist.ReduceOp) -> float:
    if is_dist_available_and_initialized() is False:
        return value

    value = to_tensor(value, device)
    dist.barrier()
    dist.all_reduce(value, op)

    return value.item()  # type: ignore[no-any-return]


def get_world_size() -> int:
    if is_dist_available_and_initialized() is False:
        return 1

    return dist.get_world_size()  # type: ignore[no-any-return]


def is_global_primary(args: argparse.Namespace) -> bool:
    return args.rank == 0  # type: ignore[no-any-return]


def is_local_primary(args: argparse.Namespace) -> bool:
    if is_dist_available_and_initialized() is False:
        return True

    return args.local_rank == 0  # type: ignore[no-any-return]


###############################################################################
# Utility Functions
###############################################################################


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    warmup_epochs: float,
    iter_per_epoch: int,
    start_warmup_value: float = 0.0,
) -> list[float]:
    """
    Create a learning rate schedule with linear warmup followed by cosine annealing

    The schedule consists of two phases:
    1. Linear warmup: linearly increases from start_warmup_value to base_value
    2. Cosine annealing: follows cosine decay from base_value to final_value

    Parameters
    ----------
    base_value
        Peak value after warmup phase (start of cosine decay).
    final_value
        Final value at the end of training.
    epochs
        Total number of training epochs.
    warmup_epochs
        Number of epochs for warmup phase. Can be fractional (e.g., 0.5 for half epoch).
    iter_per_epoch
        Number of iterations per epoch.
    start_warmup_value
        Starting value for warmup phase.

    Returns
    -------
    List of scheduler values for each iteration. Length equals epochs * iter_per_epoch.
    """

    if warmup_epochs >= epochs:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be less than epochs ({epochs})")
    if epochs <= 0 or iter_per_epoch <= 0:
        raise ValueError("epochs and iter_per_epoch must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")

    total_iterations = epochs * iter_per_epoch
    warmup_iterations = int(warmup_epochs * iter_per_epoch)
    cosine_iterations = total_iterations - warmup_iterations

    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iterations, endpoint=False)
    iters = np.arange(cosine_iterations)
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    assert len(schedule) == total_iterations

    return schedule.tolist()  # type: ignore[no-any-return]


def scale_lr(args: argparse.Namespace) -> float:
    lr: float = args.lr
    logger.debug(f"Learning rate before scaling: {lr}")
    if args.lr_scale is not None:
        ratio = args.batch_size * args.grad_accum_steps * args.world_size / args.lr_scale
        if args.lr_scale_type == "sqrt":
            ratio = ratio**0.5

        lr = lr * ratio
        logger.info(f"Adjusted learning rate to: {lr}")

    return lr


def log_git_info() -> None:
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command: list[str]) -> str:
        try:
            return (
                subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL)  # nosec # only hard-coded args
                .decode("ascii")
                .strip()
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return "N/A"

    sha = _run(["git", "rev-parse", "HEAD"])
    status = "unclean" if _run(["git", "diff-index", "HEAD"]) else "clean"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    logger.info(f"[GIT] sha: {sha}, status: {status}, branch: {branch}")


def training_log_name(network: str, device: torch.device) -> str:
    timestamp = datetime.now().replace(microsecond=0)
    if is_dist_available_and_initialized() is True:
        posix_ts = timestamp.timestamp()
        posix_ts_t = torch.tensor(posix_ts, dtype=torch.float64, device=device)
        dist.broadcast(posix_ts_t, src=0, async_op=False)
        posix_ts = posix_ts_t.item()
        timestamp = datetime.fromtimestamp(posix_ts)

    iso_timestamp = timestamp.strftime("%Y-%m-%dT%H%M%S")
    return f"{network}__{iso_timestamp}"


def training_log_path(network_name: str, device: torch.device, experiment: Optional[str]) -> Path:
    log_name = training_log_name(network_name, device)
    if experiment is not None:
        log_dir = settings.TRAINING_RUNS_PATH.joinpath(experiment)
        log_dir.mkdir(exist_ok=True)
    else:
        log_dir = settings.TRAINING_RUNS_PATH

    return log_dir.joinpath(log_name)


def setup_file_logging(log_file_path: str | Path) -> logging.Handler:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        fmt="{message}",
        style="{",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.LOG_LEVEL)

    birder_logger = logging.getLogger("birder")
    birder_logger.addHandler(file_handler)

    return file_handler


@contextlib.contextmanager
def single_handler_logging(
    target_logger: logging.Logger, target_handler: logging.Handler, enabled: bool = True
) -> Generator[logging.Logger, None, None]:
    """
    Context manager to temporarily use only a specific handler for logging.

    Parameters
    ----------
    target_logger
        The logger to modify.
    target_handler
        The handler to keep (all others will be temporarily removed).
    enabled
        If False, acts as a no-op and yields the original logger unchanged.

    Yields
    ------
    The logger object, either modified to use only the target handler
    (if enabled=True) or unchanged (if enabled=False).

    Examples
    --------
    >>> with single_handler_logging(logger, file_handler, enabled=file_only_logging) as log:
    ...     log.info("This only goes to the target handler")
    """

    if enabled is False:
        yield target_logger
        return

    # Store original state
    original_handlers = list(target_logger.handlers)
    original_propagate = target_logger.propagate

    # Remove all handlers
    for handler in original_handlers:
        target_logger.removeHandler(handler)

    target_logger.addHandler(target_handler)

    # Disable propagation to parent loggers
    target_logger.propagate = False

    try:
        yield target_logger
    finally:
        # Restore all handlers in the original order
        target_logger.removeHandler(target_handler)
        for handler in original_handlers:
            target_logger.addHandler(handler)

        target_logger.propagate = original_propagate


def get_grad_norm(parameters: Iterator[torch.Tensor], norm_type: float = 2.0) -> float:
    filtered_parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm: float = 0.0
    for p in filtered_parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    return total_norm


def freeze_batchnorm2d(module: torch.nn.Module) -> torch.nn.Module:
    """
    Referenced from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/norm_act.py#L251
    """

    res = module
    if isinstance(module, (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine is True:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()

        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = freeze_batchnorm2d(child)
            if new_child is not child:
                res.add_module(name, new_child)

    return res


def replace_module(
    module: torch.nn.Module,
    from_module: type[torch.nn.Module] | tuple[type[torch.nn.Module], ...],
    to_module: type[torch.nn.Module],
) -> torch.nn.Module:
    res = module
    if isinstance(module, from_module):
        res = to_module()
    else:
        for name, child in module.named_children():
            new_child = replace_module(child, from_module, to_module)
            if new_child is not child:
                res.add_module(name, new_child)

    return res
