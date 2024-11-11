import warnings
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load

import birder


def load_msda() -> Optional[ModuleType]:
    if torch.cuda.is_available() is False:
        return None

    # Adapted from:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/load_custom.py
    root = Path(birder.__file__).resolve().parent.joinpath("kernels/deformable_detr")
    src_files = [
        root.joinpath("vision.cpp"),
        root.joinpath("cpu/ms_deform_attn_cpu.cpp"),
        root.joinpath("cuda/ms_deform_attn_cuda.cu"),
    ]

    with warnings.catch_warnings(action="ignore", category=UserWarning):
        msda: ModuleType = load(
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

    return msda
