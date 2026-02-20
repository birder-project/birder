import argparse
import logging
import re
from collections.abc import Sequence
from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import OffloadPolicy
from torch.distributed.fsdp import fully_shard

logger = logging.getLogger(__name__)


def is_fsdp_mode(args: argparse.Namespace) -> bool:
    if dist.is_available() is False or dist.is_initialized() is False:
        return False

    return getattr(args, "distributed_mode", "ddp") == "fsdp"


def _reshard_after_forward(strategy: str) -> bool:
    if strategy == "shard-grad-op":
        return False
    if strategy == "full-shard":
        return True

    raise ValueError(f"Unsupported FSDP sharding strategy: {strategy}")


def modules_from_block_group_regex(module: torch.nn.Module, block_group_regex: str) -> list[torch.nn.Module]:
    pattern = re.compile(block_group_regex)
    name_to_module = dict(module.named_modules())
    matched_modules: list[torch.nn.Module] = []
    seen_names: set[str] = set()
    for parameter_name, _ in module.named_parameters():
        match = pattern.search(parameter_name)
        if match is None:
            continue

        module_name = parameter_name[: match.end()]
        wrap_module = name_to_module.get(module_name)
        if wrap_module is None or module_name in seen_names:
            continue

        seen_names.add(module_name)
        matched_modules.append(wrap_module)
        logger.debug(f"FSDP wrap module added (block-group-regex): {module_name}")

    return matched_modules


def modules_from_stages(module: torch.nn.Module) -> list[torch.nn.Module]:
    return_stages = getattr(module, "return_stages", None)
    if return_stages is None:
        return []

    stage_names = ["stem", *return_stages]
    name_to_module = dict(module.named_modules())
    matched_modules: list[torch.nn.Module] = []
    seen_module_ids: set[int] = set()
    for stage_name in stage_names:
        resolved_name = stage_name
        stage_module = name_to_module.get(stage_name)
        if stage_module is None:
            resolved_name = f"body.{stage_name}"
            stage_module = name_to_module.get(resolved_name)

        if stage_module is None:
            continue

        module_id = id(stage_module)
        if module_id in seen_module_ids:
            continue

        seen_module_ids.add(module_id)
        matched_modules.append(stage_module)
        logger.debug(f"FSDP wrap module added (stages): {resolved_name}")

    return matched_modules


def modules_from_min_num_params(module: torch.nn.Module, min_num_params: int) -> list[torch.nn.Module]:
    matched_modules: list[torch.nn.Module] = []
    covered_parameter_ids: set[int] = set()

    # Traverse bottom-up and pick non-overlapping module groups by uncovered parameter count
    for module_name, candidate_module in reversed(list(module.named_modules())):
        if module_name == "":
            continue

        candidate_parameters = list(candidate_module.parameters())
        if len(candidate_parameters) == 0:
            continue

        uncovered_parameters = [p for p in candidate_parameters if id(p) not in covered_parameter_ids]
        uncovered_numel = sum(p.numel() for p in uncovered_parameters)
        if uncovered_numel < min_num_params:
            continue

        matched_modules.append(candidate_module)
        covered_parameter_ids.update(id(p) for p in candidate_parameters)
        logger.debug(f"FSDP wrap module added (min-num-params): {module_name} ({uncovered_numel:,} uncovered params)")

    return matched_modules


def setup_fsdp(
    net: torch.nn.Module,
    args: argparse.Namespace,
    wrap_modules: Optional[Sequence[torch.nn.Module]] = None,
    mesh: Optional[DeviceMesh] = None,
) -> FSDPModule:
    sync_module_states(net)
    param_dtype = None if args.fsdp_param_dtype is None else getattr(torch, args.fsdp_param_dtype)
    reduce_dtype = None if args.fsdp_reduce_dtype is None else getattr(torch, args.fsdp_reduce_dtype)
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    if args.fsdp_offload_policy == "none":
        offload_policy: OffloadPolicy = OffloadPolicy()
    elif args.fsdp_offload_policy == "cpu":
        offload_policy = CPUOffloadPolicy()
    else:
        raise ValueError(f"Unsupported FSDP offload policy: {args.fsdp_offload_policy}")

    reshard_after_forward = _reshard_after_forward(args.fsdp_sharding_strategy)
    if mesh is None:
        mesh = init_device_mesh("cuda", (args.world_size,), mesh_dim_names=("dp",))

    if wrap_modules is None:
        modules_to_wrap: Sequence[torch.nn.Module] = ()
    else:
        modules_to_wrap = wrap_modules

    for module in modules_to_wrap:
        fully_shard(
            module,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )

    net = fully_shard(
        net, mesh=mesh, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy, offload_policy=offload_policy
    )

    return net


@torch.no_grad()  # type: ignore[untyped-decorator]
def sync_module_states(net: torch.nn.Module, src: int = 0) -> None:
    for parameter in net.parameters():
        if parameter.device.type == "meta":
            continue

        dist.broadcast(parameter.detach(), src=src)

    for buffer in net.buffers():
        if buffer.device.type == "meta":
            continue

        dist.broadcast(buffer.detach(), src=src)


@torch.no_grad()  # type: ignore[untyped-decorator]
def broadcast_module_buffers(net: torch.nn.Module, src: int = 0) -> None:
    for buffer in net.buffers():
        if buffer.device.type == "meta":
            continue

        dist.broadcast(buffer, src=src)


def extract_submodule_state_dict(
    full_state_dict: dict[str, Any], parent: torch.nn.Module, submodule: torch.nn.Module
) -> dict[str, Any]:
    for name, mod in parent.named_modules():
        if mod is submodule:
            prefix = f"{name}."
            prefix_len = len(prefix)
            return {k[prefix_len:]: v for k, v in full_state_dict.items() if k.startswith(prefix)}

    raise ValueError("submodule not found in parent")


def gather_full_model_state_dict(
    net: torch.nn.Module, *, cpu_offload: bool = True, broadcast_from_rank0: bool = False
) -> dict[str, Any]:
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=broadcast_from_rank0)
    return get_model_state_dict(net, options=options)  # type: ignore[no-any-return]


def gather_full_optimizer_state_dict(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer, cpu_offload: bool = True, broadcast_from_rank0: bool = False
) -> dict[str, Any]:
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=broadcast_from_rank0)
    return get_optimizer_state_dict(net, optimizer, options=options)  # type: ignore[no-any-return]


def load_full_optimizer_state_dict(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
    broadcast_from_rank0: bool = False,
) -> None:
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=broadcast_from_rank0)
    set_optimizer_state_dict(net, optimizer, optimizer_state, options=options)
