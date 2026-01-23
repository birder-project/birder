"""
RoPE ViT model configuration registrations

This file contains *only* model variant definitions and their registration
with the global model registry. The actual RoPE ViT implementation lives in rope_vit.py.
"""

from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import GIANT
from birder.net._vit_configs import GIGANTIC
from birder.net._vit_configs import HUGE
from birder.net._vit_configs import LARGE
from birder.net._vit_configs import MEDIUM
from birder.net._vit_configs import SMALL
from birder.net._vit_configs import SO150
from birder.net._vit_configs import SO400
from birder.net._vit_configs import TINY
from birder.net.base import BaseNet

# Vision Transformer Model Naming Convention
# ==========================================
#
# Model names follow a structured pattern to encode architectural choices:
# [rope_]vit_[reg{N}_][size][patch_size][_components][_pooling][_c{N}]
#
# Core Components:
# - rope_       : Rotary Position Embedding (RoPE) enabled
# - rope_i_     : Rotary Position Embedding (RoPE) enabled with interleaved rotation - implies different temp, indexing
# - vit_        : Vision Transformer base architecture
# - reg{N}_     : Register tokens (N = number of register tokens, e.g., reg4, reg8)
# - size        : Model size (s=small, b=base, l=large, or specific like so150m)
# - patch_size  : Patch size (e.g., 14, 16, 32 for 14x14, 16x16, 32x32 patches)
#
# Optional Components:
#     Position Embeddings:
#     - nps         : No Position embedding on Special tokens
#
#     Normalization:
#     - rms         : RMSNorm (instead of LayerNorm)
#     - pn          : Pre-Norm (layer norm before the encoder) - implies norm eps of 1e-5
#     - npn         : No Post Norm (disables post-normalization layer)
#     - qkn         : QK Norm
#
#     Feed-Forward Network:
#     - swiglu      : SwiGLU FFN layer type (instead of standard FFN)
#
#     Activation:
#     - quick_gelu  : QuickGELU activation type
#     - ...
#
#     Regularization:
#     - ls          : Layer Scaling applied
#
#     Pooling/Reduction:
#     - avg         : Average pooling for sequence reduction
#     - ap          : Attention Pooling for sequence reduction
#     - aps         : Attention Pooling inc. Special tokens for sequence reduction
#
#     Custom Variants:
#     - c{N}        : Custom variant (N = version number) for models with fine-grained or non-standard
#                     modifications not fully reflected in the name


def register_rope_vit_configs(rope_vit: type[BaseNet]) -> None:
    registry.register_model_config(
        "rope_vit_t32",
        rope_vit,
        config={"patch_size": 32, **TINY},
    )
    registry.register_model_config(
        "rope_vit_t16",
        rope_vit,
        config={"patch_size": 16, **TINY},
    )
    registry.register_model_config(
        "rope_vit_t14",
        rope_vit,
        config={"patch_size": 14, **TINY},
    )
    registry.register_model_config(
        "rope_vit_s32",
        rope_vit,
        config={"patch_size": 32, **SMALL},
    )
    registry.register_model_config(
        "rope_vit_s16",
        rope_vit,
        config={"patch_size": 16, **SMALL},
    )
    registry.register_model_config(
        "rope_vit_s16_avg",
        rope_vit,
        config={"patch_size": 16, **SMALL, "class_token": False},
    )
    registry.register_model_config(
        "rope_i_vit_s16_pn_aps_c1",  # For PE Core - https://arxiv.org/abs/2504.13181
        rope_vit,
        config={
            "patch_size": 16,
            **SMALL,
            "pre_norm": True,
            "attn_pool_head": True,
            "attn_pool_num_heads": 8,
            "attn_pool_special_tokens": True,
            "norm_layer_eps": 1e-5,
            "rope_rot_type": "interleaved",
            "rope_grid_indexing": "xy",
            "rope_grid_offset": 1,
            "rope_temperature": 10000.0,
        },
    )
    registry.register_model_config(
        "rope_vit_s14",
        rope_vit,
        config={"patch_size": 14, **SMALL},
    )
    registry.register_model_config(
        "rope_vit_m32",
        rope_vit,
        config={"patch_size": 32, **MEDIUM},
    )
    registry.register_model_config(
        "rope_vit_m16",
        rope_vit,
        config={"patch_size": 16, **MEDIUM},
    )
    registry.register_model_config(
        "rope_vit_m14",
        rope_vit,
        config={"patch_size": 14, **MEDIUM},
    )
    registry.register_model_config(
        "rope_vit_b32",
        rope_vit,
        config={"patch_size": 32, **BASE, "drop_path_rate": 0.0},  # Override the BASE definition
    )
    registry.register_model_config(
        "rope_vit_b16",
        rope_vit,
        config={"patch_size": 16, **BASE},
    )
    registry.register_model_config(
        "rope_vit_b16_qkn_ls",
        rope_vit,
        config={"patch_size": 16, **BASE, "layer_scale_init_value": 1e-5, "qk_norm": True},
    )
    registry.register_model_config(
        "rope_i_vit_b16_pn_aps_c1",  # For PE Core - https://arxiv.org/abs/2504.13181
        rope_vit,
        config={
            "patch_size": 16,
            **BASE,
            "pre_norm": True,
            "attn_pool_head": True,
            "attn_pool_num_heads": 8,
            "attn_pool_special_tokens": True,
            "norm_layer_eps": 1e-5,
            "rope_rot_type": "interleaved",
            "rope_grid_indexing": "xy",
            "rope_grid_offset": 1,
            "rope_temperature": 10000.0,
        },
    )
    registry.register_model_config(
        "rope_vit_b14",
        rope_vit,
        config={"patch_size": 14, **BASE},
    )
    registry.register_model_config(
        "rope_vit_so150m_p14_ap",
        rope_vit,
        config={"patch_size": 14, **SO150, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "rope_vit_l32",
        rope_vit,
        config={"patch_size": 32, **LARGE},
    )
    registry.register_model_config(
        "rope_vit_l16",
        rope_vit,
        config={"patch_size": 16, **LARGE},
    )
    registry.register_model_config(
        "rope_vit_l14",
        rope_vit,
        config={"patch_size": 14, **LARGE},
    )
    registry.register_model_config(
        "rope_i_vit_l14_pn_aps_c1",  # For PE Core - https://arxiv.org/abs/2504.13181
        rope_vit,
        config={
            "patch_size": 14,
            **LARGE,
            "pre_norm": True,
            "attn_pool_head": True,
            "attn_pool_num_heads": 8,
            "attn_pool_special_tokens": True,
            "norm_layer_eps": 1e-5,
            "rope_rot_type": "interleaved",
            "rope_grid_indexing": "xy",
            "rope_grid_offset": 1,
            "rope_temperature": 10000.0,
        },
    )
    registry.register_model_config(
        "rope_vit_so400m_p14_ap",
        rope_vit,
        config={"patch_size": 14, **SO400, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "rope_vit_h16",
        rope_vit,
        config={"patch_size": 16, **HUGE},
    )
    registry.register_model_config(
        "rope_vit_h14",
        rope_vit,
        config={"patch_size": 14, **HUGE},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "rope_vit_g16",
        rope_vit,
        config={"patch_size": 16, **GIANT},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "rope_vit_g14",
        rope_vit,
        config={"patch_size": 14, **GIANT},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "rope_vit_gigantic14",
        rope_vit,
        config={"patch_size": 14, **GIGANTIC},
    )

    # With registers
    ####################

    registry.register_model_config(
        "rope_vit_reg1_t32",
        rope_vit,
        config={"patch_size": 32, **TINY, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "rope_vit_reg1_t16",
        rope_vit,
        config={"patch_size": 16, **TINY, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "rope_vit_reg1_t14",
        rope_vit,
        config={"patch_size": 14, **TINY, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "rope_vit_reg1_s32",
        rope_vit,
        config={"patch_size": 32, **SMALL, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "rope_vit_reg1_s16",
        rope_vit,
        config={"patch_size": 16, **SMALL, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "rope_i_vit_reg1_s16_pn_npn_avg_c1",  # For PE Spatial - https://arxiv.org/abs/2504.13181
        rope_vit,
        config={
            "patch_size": 16,
            **SMALL,
            "num_reg_tokens": 1,
            "class_token": False,
            "pre_norm": True,
            "post_norm": False,
            "norm_layer_eps": 1e-5,
            "rope_rot_type": "interleaved",
            "rope_grid_indexing": "xy",
            "rope_grid_offset": 1,
            "rope_temperature": 10000.0,
        },
    )
    registry.register_model_config(
        "rope_vit_reg1_s14",
        rope_vit,
        config={"patch_size": 14, **SMALL, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "rope_vit_reg4_m32",
        rope_vit,
        config={"patch_size": 32, **MEDIUM, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_m16",
        rope_vit,
        config={"patch_size": 16, **MEDIUM, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_m16_rms_avg",
        rope_vit,
        config={"patch_size": 16, **MEDIUM, "num_reg_tokens": 4, "class_token": False, "norm_layer_type": "RMSNorm"},
    )
    registry.register_model_config(
        "rope_vit_reg4_m14",
        rope_vit,
        config={"patch_size": 14, **MEDIUM, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_m14_avg",
        rope_vit,
        config={"patch_size": 14, **MEDIUM, "num_reg_tokens": 4, "class_token": False},
    )
    registry.register_model_config(
        "rope_vit_reg4_b32",
        rope_vit,
        config={"patch_size": 32, **BASE, "num_reg_tokens": 4, "drop_path_rate": 0.0},  # Override the BASE definition
    )
    registry.register_model_config(
        "rope_vit_reg4_b16",
        rope_vit,
        config={"patch_size": 16, **BASE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_b14",
        rope_vit,
        config={"patch_size": 14, **BASE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg8_nps_b14_ap",
        rope_vit,
        config={
            "pos_embed_special_tokens": False,
            "patch_size": 14,
            **BASE,
            "num_reg_tokens": 8,
            "class_token": False,
            "attn_pool_head": True,
        },
    )
    registry.register_model_config(
        "rope_vit_reg4_so150m_p14_ap",
        rope_vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 4, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "rope_vit_reg8_so150m_p14_ap",
        rope_vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "rope_vit_reg8_so150m_p14_swiglu_rms_avg",
        rope_vit,
        config={
            "patch_size": 14,
            **SO150,
            "num_reg_tokens": 8,
            "class_token": False,
            "norm_layer_type": "RMSNorm",
            "mlp_layer_type": "SwiGLU_FFN",
        },
    )
    registry.register_model_config(
        "rope_vit_reg8_so150m_p14_swiglu_rms_ap",
        rope_vit,
        config={
            "patch_size": 14,
            **SO150,
            "num_reg_tokens": 8,
            "class_token": False,
            "attn_pool_head": True,
            "norm_layer_type": "RMSNorm",
            "mlp_layer_type": "SwiGLU_FFN",
        },
    )
    registry.register_model_config(
        "rope_vit_reg8_so150m_p14_swiglu_rms_aps",
        rope_vit,
        config={
            "patch_size": 14,
            **SO150,
            "num_reg_tokens": 8,
            "class_token": False,
            "attn_pool_head": True,
            "attn_pool_special_tokens": True,
            "norm_layer_type": "RMSNorm",
            "mlp_layer_type": "SwiGLU_FFN",
        },
    )
    registry.register_model_config(
        "rope_vit_reg4_l32",
        rope_vit,
        config={"patch_size": 32, **LARGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_l16",
        rope_vit,
        config={"patch_size": 16, **LARGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_l14",
        rope_vit,
        config={"patch_size": 14, **LARGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg8_l14_rms_ap",
        rope_vit,
        config={
            "patch_size": 14,
            **LARGE,
            "num_reg_tokens": 8,
            "class_token": False,
            "attn_pool_head": True,
            "norm_layer_type": "RMSNorm",
        },
    )
    registry.register_model_config(
        "rope_vit_reg8_so400m_p14_ap",
        rope_vit,
        config={"patch_size": 14, **SO400, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "rope_vit_reg4_h16",
        rope_vit,
        config={"patch_size": 16, **HUGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "rope_vit_reg4_h14",
        rope_vit,
        config={"patch_size": 14, **HUGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "rope_vit_reg4_g14",
        rope_vit,
        config={"patch_size": 14, **GIANT, "num_reg_tokens": 4},
    )
