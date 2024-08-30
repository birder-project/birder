import argparse
import logging
import math
import os
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from typing import Literal
from typing import Optional
from typing import Sized

import torch
import torch.amp
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

OptimizerType = Literal["sgd", "rmsprop", "adamw"]
SchedulerType = Literal["constant", "step", "cosine"]


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


# pylint: disable=protected-access,too-many-locals,too-many-branches
def optimizer_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    custom_keys_weight_decay: Optional[list[tuple[str, float]]] = None,
    layer_decay: Optional[float] = None,
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
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    # Count layers
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
                logging.info("Found duplicated parameters (probably a module alias)")
                user_warned = True

            skip_module = True

        visited_modules.append(id(module))
        parameters_found = False
        for name, p in module.named_parameters(recurse=False):
            if skip_module is True:
                break

            parameters_found = True
            if p.requires_grad is False:
                continue

            is_custom_key = False
            if custom_keys_weight_decay is not None:
                for key, custom_wd in custom_keys_weight_decay:
                    target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                    if key == target_name:
                        params.append(
                            {
                                "params": p,
                                "weight_decay": custom_wd,
                                "lr_scale": 1.0 if layer_decay is None else layer_scales[idx],
                            }
                        )
                        is_custom_key = True
                        break

            if is_custom_key is False:
                if norm_weight_decay is not None and isinstance(module, norm_classes) is True:
                    params.append(
                        {
                            "params": p,
                            "weight_decay": norm_weight_decay,
                            "lr_scale": 1.0 if layer_decay is None else layer_scales[idx],
                        }
                    )

                else:
                    params.append(
                        {
                            "params": p,
                            "weight_decay": weight_decay,
                            "lr_scale": 1.0 if layer_decay is None else layer_scales[idx],
                        }
                    )

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
            "position_embedding",
            "relative_position_bias_table",
            "decoder_embed",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    return custom_keys_weight_decay


def get_optimizer(
    opt: OptimizerType,
    parameters: list[dict[str, Any]],
    lr: float,
    wd: float,
    momentum: float,
    nesterov: bool,
) -> torch.optim.Optimizer:
    if opt == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd)
    elif opt == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=wd, eps=0.0316, alpha=0.9)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
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
    lr_step_gamma: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    # Warmup epochs is given in absolute number from 0
    remaining_warmup = max(0, warmup_epochs - begin_epoch)
    if lr_scheduler == "constant":
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    elif lr_scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    elif lr_scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - begin_epoch + 1 - remaining_warmup), eta_min=lr_cosine_min
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
        if amp_dtype_str == "float16":
            amp_dtype = torch.float16
        elif amp_dtype_str == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown dtype {amp_dtype_str}")

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


def init_distributed_mode(args: argparse.Namespace) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    else:
        logging.info("Not using distributed mode")
        args.rank = 0
        args.distributed = False
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    logging.info(f"Distributed init (rank {args.rank}): {args.dist_url}")
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    if args.rank != 0:
        disable_print()
        logging.disable(logging.CRITICAL)


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


def reduce_across_processes(value: float, device: torch.device) -> float:
    if is_dist_available_and_initialized() is False:
        return value

    value_t = torch.tensor(value, device=device)
    dist.barrier()
    dist.all_reduce(value_t)

    return value_t.item()  # type: ignore


def training_log_name(network: str, device: torch.device) -> str:
    timestamp = datetime.now().replace(microsecond=0)
    if is_dist_available_and_initialized() is True:
        posix_ts = timestamp.timestamp()
        posix_ts_t = torch.tensor(posix_ts, dtype=torch.float64, device=device)
        dist.broadcast(posix_ts_t, src=0, async_op=False)
        posix_ts = posix_ts_t.item()
        timestamp = datetime.fromtimestamp(posix_ts)

    iso_timestamp = timestamp.isoformat()
    return f"{network}__{iso_timestamp}"


def get_grad_norm(parameters: Iterator[torch.Tensor], norm_type: float = 2) -> float:
    filtered_parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0.0
    for p in filtered_parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    return total_norm
