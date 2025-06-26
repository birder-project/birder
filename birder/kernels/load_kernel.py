import logging
import os
import warnings
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load

import birder

logger = logging.getLogger(__name__)


_CACHED_KERNELS: dict[str, ModuleType] = {}


def load_msda() -> Optional[ModuleType]:
    name = "msda"
    if torch.cuda.is_available() is False or os.environ.get("DISABLE_CUSTOM_KERNELS", "0") == "1":
        return None

    if name in _CACHED_KERNELS:
        return _CACHED_KERNELS[name]

    # Adapted from:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/load_custom.py
    root = Path(birder.__file__).resolve().parent.joinpath("kernels/deformable_detr")
    src_files = [
        root.joinpath("vision.cpp"),
        root.joinpath("cpu/ms_deform_attn_cpu.cpp"),
        root.joinpath("cuda/ms_deform_attn_cuda.cu"),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        msda: Optional[ModuleType] = load(
            "MultiScaleDeformableAttention",
            src_files,
            with_cuda=True,
            extra_include_paths=[str(root)],
            extra_cflags=["-DWITH_CUDA=1"],
            extra_cuda_cflags=[
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        )

    if msda is not None:
        logger.info("MSDA custom kernel loaded")
        _CACHED_KERNELS[name] = msda
    else:
        logger.debug("MSDA custom kernel NOT loaded")

    return msda


def load_swattention() -> Optional[ModuleType]:
    name = "swattention"
    if torch.cuda.is_available() is False or os.environ.get("DISABLE_CUSTOM_KERNELS", "0") == "1":
        return None

    if name in _CACHED_KERNELS:
        return _CACHED_KERNELS[name]

    root = Path(birder.__file__).resolve().parent.joinpath("kernels/transnext")
    src_files = [
        root.joinpath("swattention.cpp"),
        root.joinpath("av_bw_kernel.cu"),
        root.joinpath("av_fw_kernel.cu"),
        root.joinpath("qk_bw_kernel.cu"),
        root.joinpath("qk_fw_kernel.cu"),
        root.joinpath("qk_rpb_bw_kernel.cu"),
        root.joinpath("qk_rpb_fw_kernel.cu"),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        swattention: Optional[ModuleType] = load(
            "swattention",
            src_files,
            with_cuda=True,
            extra_cflags=["-DWITH_CUDA=1"],
            extra_cuda_cflags=[
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        )

    if swattention is not None:
        logger.info("swattention custom kernel loaded")
        _CACHED_KERNELS[name] = swattention
    else:
        logger.debug("swattention custom kernel NOT loaded")

    return swattention


def load_soft_nms() -> Optional[ModuleType]:
    name = "soft_nms"
    if os.environ.get("DISABLE_CUSTOM_KERNELS", "0") == "1":
        return None

    if name in _CACHED_KERNELS:
        return _CACHED_KERNELS[name]

    root = Path(birder.__file__).resolve().parent.joinpath("kernels/soft_nms")
    src_files = [
        root.joinpath("op.cpp"),
        root.joinpath("soft_nms.cpp"),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        soft_nms: Optional[ModuleType] = load(
            "soft_nms",
            src_files,
            with_cuda=True,
            extra_cflags=["-DWITH_CUDA=1"],
            extra_cuda_cflags=[
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        )

    if soft_nms is not None:
        logger.info("soft_nms custom kernel loaded")
        _CACHED_KERNELS[name] = soft_nms
    else:
        logger.debug("soft_nms custom kernel NOT loaded")

    return soft_nms
