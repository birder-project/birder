import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load

import birder

logger = logging.getLogger(__name__)


_KERNELS_DIR = Path(birder.__file__).resolve().parent.joinpath("kernels")
_COMMON_CUDA_FLAGS = (
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
)


@dataclass(frozen=True, slots=True)
class KernelSpec:
    key: str
    extension_name: str
    directory: str
    sources: tuple[str, ...]
    with_cuda: bool = False
    cuda_only: bool = False
    extra_include_paths: tuple[str, ...] = ()
    extra_cflags: tuple[str, ...] = ()
    extra_cuda_cflags: tuple[str, ...] = ()


_LINEAR_ASSIGNMENT_SPEC = KernelSpec(
    key="linear_assignment",
    extension_name="linear_assignment",
    directory="linear_assignment",
    sources=(
        "op.cpp",
        "linear_assignment_cuda.cu",
    ),
    with_cuda=True,
    cuda_only=True,
    extra_cflags=("-O3",),
    extra_cuda_cflags=("-O3", "--restrict"),
)
_MSDA_SPEC = KernelSpec(
    key="msda",
    extension_name="MultiScaleDeformableAttention",
    directory="msda",
    sources=(
        "op.cpp",
        "cuda/ms_deform_attn_cuda.cu",
    ),
    with_cuda=True,
    cuda_only=True,
    extra_cflags=("-O3",),
    extra_cuda_cflags=("-O3", "--restrict", *_COMMON_CUDA_FLAGS),
)
_SOFT_NMS_SPEC = KernelSpec(
    key="soft_nms",
    extension_name="soft_nms",
    directory="soft_nms",
    sources=(
        "op.cpp",
        "soft_nms_cuda.cu",
    ),
    with_cuda=True,
    cuda_only=True,
    extra_cflags=("-O3",),
    extra_cuda_cflags=("-O3", "--restrict", *_COMMON_CUDA_FLAGS),
)
_SWATTENTION_SPEC = KernelSpec(
    key="swattention",
    extension_name="swattention",
    directory="swattention",
    sources=(
        "op.cpp",
        "av_bw_kernel.cu",
        "av_fw_kernel.cu",
        "qk_bw_kernel.cu",
        "qk_fw_kernel.cu",
        "qk_rpb_bw_kernel.cu",
        "qk_rpb_fw_kernel.cu",
    ),
    with_cuda=True,
    cuda_only=True,
    extra_cflags=("-O3",),
    extra_cuda_cflags=("-O3", *_COMMON_CUDA_FLAGS),
)

_CACHED_KERNELS: dict[str, Optional[ModuleType]] = {}
_DISABLED_CUSTOM_KERNELS: set[str] = set()
_CUSTOM_KERNELS_ENABLED = True


def set_custom_kernels_enabled(enabled: bool) -> None:
    global _CUSTOM_KERNELS_ENABLED  # pylint: disable=global-statement
    _CUSTOM_KERNELS_ENABLED = enabled


def set_custom_kernel_enabled(kernel: str, enabled: bool) -> None:
    """
    Enable or disable loading of an individual custom kernel
    """

    if enabled is True:
        _DISABLED_CUSTOM_KERNELS.discard(kernel)
    else:
        _DISABLED_CUSTOM_KERNELS.add(kernel)


def is_custom_kernels_enabled(kernel: Optional[str] = None) -> bool:
    if os.environ.get("DISABLE_CUSTOM_KERNELS", "0") == "1":
        return False

    if kernel is not None:
        if kernel in _DISABLED_CUSTOM_KERNELS:
            return False

        if os.environ.get(f"DISABLE_CUSTOM_KERNELS_{kernel.upper()}", "0") == "1":
            return False

    return _CUSTOM_KERNELS_ENABLED


def _load_kernel(spec: KernelSpec) -> Optional[ModuleType]:
    if (spec.cuda_only is True and torch.cuda.is_available() is False) or is_custom_kernels_enabled(spec.key) is False:
        return None

    if spec.key in _CACHED_KERNELS:
        return _CACHED_KERNELS[spec.key]

    root = _KERNELS_DIR.joinpath(spec.directory)
    source_files = [root.joinpath(source) for source in spec.sources]
    extra_include_paths = [str(root.joinpath(path)) for path in spec.extra_include_paths]

    kernel: Optional[ModuleType]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            library_path = load(
                spec.extension_name,
                source_files,
                with_cuda=spec.with_cuda,
                extra_include_paths=extra_include_paths,
                extra_cflags=list(spec.extra_cflags),
                extra_cuda_cflags=list(spec.extra_cuda_cflags),
                is_python_module=False,
            )

        ops_namespace = Path(library_path).stem
        kernel = getattr(torch.ops, ops_namespace, None)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(f"{spec.key} custom kernel failed to load, using fallback")
        _CACHED_KERNELS[spec.key] = None
        return None

    _CACHED_KERNELS[spec.key] = kernel
    if kernel is not None:
        logger.info(f"{spec.key} custom kernel loaded")
    else:
        logger.debug(f"{spec.key} custom kernel NOT loaded")

    return kernel


def load_linear_assignment() -> Optional[ModuleType]:
    return _load_kernel(_LINEAR_ASSIGNMENT_SPEC)


def load_msda() -> Optional[ModuleType]:
    return _load_kernel(_MSDA_SPEC)


def load_soft_nms() -> Optional[ModuleType]:
    return _load_kernel(_SOFT_NMS_SPEC)


def load_swattention() -> Optional[ModuleType]:
    return _load_kernel(_SWATTENTION_SPEC)
