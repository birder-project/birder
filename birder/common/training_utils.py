import argparse
import logging
import math
import os
import re
from collections import deque
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional
from typing import Sized
from typing import get_args

import numpy as np
import torch
import torch.amp
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torchvision.ops import FrozenBatchNorm2d

from birder.optim import Lamb
from birder.optim import Lars

logger = logging.getLogger(__name__)

OptimizerType = Literal["sgd", "rmsprop", "adam", "adamw", "lamb", "lambw", "lars"]
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
    current_block: Optional[str] = None

    for s in strings:
        match = re.search(pattern, s)
        if match is not None:
            block_num = match.group(1)
            if block_num != current_block:
                if len(current_group) > 0:
                    groups.append(current_group)

                current_group = []
                current_block = block_num

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
    backbone_lr: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Return parameter groups for optimizers with per-parameter group weight decay.

    Referenced from https://github.com/pytorch/vision/blob/main/references/classification/utils.py and from
    https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
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
    layer_scales = []
    if layer_decay is not None:
        layer_max = num_layers - 1
        layer_scales = [layer_decay ** (layer_max - i) for i in range(num_layers)]

    # Set weight decay and layer decay
    user_warned = False
    idx = 0
    params = []
    module_stack_with_prefix = [(model, "")]
    visited_modules = []
    while len(module_stack_with_prefix) > 0:
        skip_module = False
        (module, prefix) = module_stack_with_prefix.pop()
        if id(module) in visited_modules:
            if user_warned is False:
                logger.info("Found duplicated parameters (probably a module alias)")
                user_warned = True

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


def get_optimizer(parameters: list[dict[str, Any]], lr: float, args: argparse.Namespace) -> torch.optim.Optimizer:
    opt: OptimizerType = args.opt
    kwargs = {}
    if getattr(args, "opt_eps", None) is not None:
        kwargs["eps"] = args.opt_eps
    if getattr(args, "opt_betas", None) is not None:
        kwargs["betas"] = args.opt_betas
    if getattr(args, "opt_alpha", None) is not None:
        kwargs["alpha"] = args.opt_alpha

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
    lr_scheduler: SchedulerType,
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    begin_epoch: int,
    epochs: int,
    lr_cosine_min: float,
    lr_step_size: int,
    lr_steps: list[int],
    lr_step_gamma: float,
    lr_power: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    # Warmup epochs is given in absolute number from 0
    remaining_warmup = max(0, warmup_epochs - begin_epoch)
    if lr_scheduler == "constant":
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    elif lr_scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    elif lr_scheduler == "multistep":
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_step_gamma)
    elif lr_scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - begin_epoch - remaining_warmup), eta_min=lr_cosine_min
        )
    elif lr_scheduler == "polynomial":
        main_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=(epochs - begin_epoch - remaining_warmup) + 1, power=lr_power
        )
    else:
        raise ValueError("Unknown learning rate scheduler")

    # Handle warmup
    if warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01 if remaining_warmup > 0 else 1, total_iters=remaining_warmup + 1
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_scheduler],
            milestones=[remaining_warmup + 1],
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


###############################################################################
# Metrics Tracking and Reporting
###############################################################################


def to_tensor(x: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)

    return torch.tensor(x, device=device)


class SmoothedValue:
    # Adapted from: https://github.com/facebookresearch/capi/blob/main/utils.py

    def __init__(self, window_size: int = 32) -> None:
        self.window_size = window_size
        self.deque: deque[torch.Tensor | float] = deque(maxlen=window_size)
        self.total: torch.Tensor | float = 0.0
        self.count: int = 0

    def update(self, value: torch.Tensor | float) -> None:
        self.deque.append(value)
        self.count += 1
        self.total += value

    def synchronize_between_processes(self, device: torch.device) -> None:
        if is_dist_available_and_initialized() is False:
            return

        logger.debug("Synchronizing values")
        count = to_tensor(self.count, device=device).to(dtype=torch.float64).reshape(1)
        total = to_tensor(self.total, device=device).to(dtype=torch.float64).reshape(1)
        tensor_deque = torch.tensor(list(self.deque), dtype=torch.float64, device=device)
        t = torch.concat([count, total, tensor_deque], dim=0)
        dist.barrier()
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
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


###############################################################################
# Distributed Training Utilities
###############################################################################


def init_distributed_mode(args: argparse.Namespace) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    else:
        logger.info("Not using distributed mode")
        args.rank = 0
        args.distributed = False
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    logger.info(f"Distributed init (rank {args.rank}): {args.dist_url}")
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier(device_ids=[args.rank])
    if args.rank != 0:
        disable_print()
        logging.disable(logging.CRITICAL)


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


###############################################################################
# Utility Functions
###############################################################################


def cosine_scheduler(base_value: float, final_value: float, epochs: int, iter_per_epoch: int) -> list[float]:
    warmup_schedule = np.array([])

    iters = np.arange(epochs * iter_per_epoch)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * iter_per_epoch

    return schedule.tolist()  # type: ignore[return-value]


def scale_lr(args: argparse.Namespace) -> float:
    lr: float = args.lr
    if args.lr_scale is not None:
        ratio = args.batch_size * args.grad_accum_steps * args.world_size / args.lr_scale
        if args.lr_scale_type == "sqrt":
            ratio = ratio**0.5

        lr = lr * ratio
        logger.info(f"Adjusted learning rate to: {lr}")

    return lr


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


def setup_file_logging(log_file_path: str | Path) -> None:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        fmt="{message}",
        style="{",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    birder_logger = logging.getLogger("birder")
    birder_logger.addHandler(file_handler)


def get_grad_norm(parameters: Iterator[torch.Tensor], norm_type: float = 2) -> float:
    filtered_parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0.0
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


###############################################################################
# Command Line Args
###############################################################################


def add_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--opt",
        type=str,
        choices=list(get_args(OptimizerType)),
        default="sgd",
        help="optimizer to use",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer momentum")
    parser.add_argument("--nesterov", default=False, action="store_true", help="use nesterov momentum")
    parser.add_argument("--opt-eps", type=float, help="optimizer epsilon (None to use the optimizer default)")
    parser.add_argument(
        "--opt-betas", type=float, nargs="+", help="optimizer betas (None to use the optimizer default)"
    )
    parser.add_argument("--opt-alpha", type=float, help="optimizer alpha (None to use the optimizer default)")


def add_scheduler_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=list(get_args(SchedulerType)),
        default="constant",
        help="learning rate scheduler",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=40,
        metavar="N",
        help="decrease lr every step-size epochs (for step scheduler only)",
    )
    parser.add_argument(
        "--lr-steps",
        type=int,
        nargs="+",
        help="decrease lr every step-size epochs (multistep scheduler only)",
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
    parser.add_argument(
        "--lr-power",
        type=float,
        default=1.0,
        help="power of the polynomial (for polynomial scheduler only)",
    )


def add_wds_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wds", default=False, action="store_true", help="use webdataset for training")
    parser.add_argument("--wds-info", type=str, metavar="FILE", help="wds info file path")
    parser.add_argument("--wds-class-file", type=str, metavar="FILE", help="class list file")
    parser.add_argument("--wds-cache-dir", type=str, help="webdataset cache directory")
    parser.add_argument("--wds-train-size", type=int, metavar="N", help="size of the wds training set")
    parser.add_argument("--wds-val-size", type=int, metavar="N", help="size of the wds validation set")
    parser.add_argument(
        "--wds-training-split", type=str, default="training", metavar="NAME", help="wds dataset train split"
    )
    parser.add_argument(
        "--wds-val-split", type=str, default="validation", metavar="NAME", help="wds dataset validation split"
    )


def add_unsupervised_wds_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wds", default=False, action="store_true", help="use webdataset for training")
    parser.add_argument("--wds-info", type=str, metavar="FILE", help="wds info file path")
    parser.add_argument("--wds-cache-dir", type=str, help="webdataset cache directory")
    parser.add_argument("--wds-train-size", type=int, metavar="N", help="size of the wds training set")
    parser.add_argument("--wds-split", type=str, default="training", metavar="NAME", help="wds dataset split to load")
