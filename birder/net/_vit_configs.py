"""
ViT model configuration registrations

This file contains *only* model variant definitions and their registration
with the global model registry. The actual ViT implementation lives in vit.py.

Naming:
- All model names must follow the ViT / RoPE ViT naming convention documented in rope_vit_configs.py.
"""

from birder.model_registry import registry
from birder.net.base import BaseNet

TINY = {"num_layers": 12, "num_heads": 3, "hidden_dim": 192, "mlp_dim": 768, "drop_path_rate": 0.0}
SMALL = {"num_layers": 12, "num_heads": 6, "hidden_dim": 384, "mlp_dim": 1536, "drop_path_rate": 0.0}
MEDIUM = {"num_layers": 12, "num_heads": 8, "hidden_dim": 512, "mlp_dim": 2048, "drop_path_rate": 0.0}
BASE = {"num_layers": 12, "num_heads": 12, "hidden_dim": 768, "mlp_dim": 3072, "drop_path_rate": 0.1}
LARGE = {"num_layers": 24, "num_heads": 16, "hidden_dim": 1024, "mlp_dim": 4096, "drop_path_rate": 0.1}
HUGE = {"num_layers": 32, "num_heads": 16, "hidden_dim": 1280, "mlp_dim": 5120, "drop_path_rate": 0.1}

# From "Getting vit in Shape: Scaling Laws for Compute-Optimal Model Design"
# Shape-optimized vision transformer (SoViT)
SO150 = {
    "num_layers": 18,
    "num_heads": 16,
    "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
    "mlp_dim": 2320,
    "drop_path_rate": 0.1,
}
SO400 = {
    "num_layers": 27,
    "num_heads": 16,
    "hidden_dim": 1152,
    "mlp_dim": 4304,
    "drop_path_rate": 0.1,
}

# From "Scaling Vision Transformers"
GIANT = {"num_layers": 40, "num_heads": 16, "hidden_dim": 1408, "mlp_dim": 6144, "drop_path_rate": 0.1}
GIGANTIC = {"num_layers": 48, "num_heads": 16, "hidden_dim": 1664, "mlp_dim": 8192, "drop_path_rate": 0.1}


def register_vit_configs(vit: type[BaseNet]) -> None:
    registry.register_model_config(
        "vit_t32",
        vit,
        config={"patch_size": 32, **TINY},
    )
    registry.register_model_config(
        "vit_t16",
        vit,
        config={"patch_size": 16, **TINY},
    )
    registry.register_model_config(
        "vit_t14",
        vit,
        config={"patch_size": 14, **TINY},
    )
    registry.register_model_config(
        "vit_s32",
        vit,
        config={"patch_size": 32, **SMALL},
    )
    registry.register_model_config(
        "vit_s16",
        vit,
        config={"patch_size": 16, **SMALL},
    )
    registry.register_model_config(
        "vit_s16_ls",
        vit,
        config={"patch_size": 16, **SMALL, "layer_scale_init_value": 1e-5},
    )
    registry.register_model_config(
        "vit_s16_pn",
        vit,
        config={"patch_size": 16, **SMALL, "pre_norm": True, "norm_layer_eps": 1e-5},
    )
    registry.register_model_config(
        "vit_s14",
        vit,
        config={"patch_size": 14, **SMALL},
    )
    registry.register_model_config(
        "vit_m32",
        vit,
        config={"patch_size": 32, **MEDIUM},
    )
    registry.register_model_config(
        "vit_m16",
        vit,
        config={"patch_size": 16, **MEDIUM},
    )
    registry.register_model_config(
        "vit_m14",
        vit,
        config={"patch_size": 14, **MEDIUM},
    )
    registry.register_model_config(
        "vit_b32",
        vit,
        config={"patch_size": 32, **BASE, "drop_path_rate": 0.0},  # Override the BASE definition
    )
    registry.register_model_config(
        "vit_b16",
        vit,
        config={"patch_size": 16, **BASE},
    )
    registry.register_model_config(
        "vit_b16_ls",
        vit,
        config={"patch_size": 16, **BASE, "layer_scale_init_value": 1e-5},
    )
    registry.register_model_config(
        "vit_b16_qkn_ls",
        vit,
        config={"patch_size": 16, **BASE, "layer_scale_init_value": 1e-5, "qk_norm": True},
    )
    registry.register_model_config(
        "vit_b16_pn_quick_gelu",
        vit,
        config={"patch_size": 16, **BASE, "pre_norm": True, "norm_layer_eps": 1e-5, "act_layer_type": "quick_gelu"},
    )
    registry.register_model_config(
        "vit_b14",
        vit,
        config={"patch_size": 14, **BASE},
    )
    registry.register_model_config(
        "vit_so150m_p14_avg",
        vit,
        config={"patch_size": 14, **SO150, "class_token": False},
    )
    registry.register_model_config(
        "vit_so150m_p14_ap",
        vit,
        config={"patch_size": 14, **SO150, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_l32",
        vit,
        config={"patch_size": 32, **LARGE},
    )
    registry.register_model_config(
        "vit_l16",
        vit,
        config={"patch_size": 16, **LARGE},
    )
    registry.register_model_config(
        "vit_l14",
        vit,
        config={"patch_size": 14, **LARGE},
    )
    registry.register_model_config(
        "vit_l14_pn",
        vit,
        config={"patch_size": 14, **LARGE, "pre_norm": True, "norm_layer_eps": 1e-5},
    )
    registry.register_model_config(
        "vit_l14_pn_quick_gelu",
        vit,
        config={"patch_size": 14, **LARGE, "pre_norm": True, "norm_layer_eps": 1e-5, "act_layer_type": "quick_gelu"},
    )
    registry.register_model_config(
        "vit_so400m_p14_ap",
        vit,
        config={"patch_size": 14, **SO400, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_h16",
        vit,
        config={"patch_size": 16, **HUGE},
    )
    registry.register_model_config(
        "vit_h14",
        vit,
        config={"patch_size": 14, **HUGE},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "vit_g16",
        vit,
        config={"patch_size": 16, **GIANT},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "vit_g14",
        vit,
        config={"patch_size": 14, **GIANT},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "vit_gigantic14",
        vit,
        config={"patch_size": 14, **GIGANTIC},
    )
    registry.register_model_config(  # From "PaLI: A Jointly-Scaled Multilingual Language-Image Model"
        "vit_e14",
        vit,
        config={
            "patch_size": 14,
            "num_layers": 56,
            "num_heads": 16,
            "hidden_dim": 1792,
            "mlp_dim": 15360,
            "drop_path_rate": 0.1,
        },
    )
    registry.register_model_config(  # From "Scaling Language-Free Visual Representation Learning"
        "vit_1b_p16",  # AKA vit_giant2 from DINOv2
        vit,
        config={
            "patch_size": 16,
            "num_layers": 40,
            "num_heads": 24,
            "hidden_dim": 1536,
            "mlp_dim": 6144,
            "drop_path_rate": 0.1,
        },
    )

    # With registers
    ####################

    registry.register_model_config(
        "vit_reg1_t32",
        vit,
        config={"patch_size": 32, **TINY, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg1_t16",
        vit,
        config={"patch_size": 16, **TINY, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg1_t14",
        vit,
        config={"patch_size": 14, **TINY, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg1_s32",
        vit,
        config={"patch_size": 32, **SMALL, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg1_s16",
        vit,
        config={"patch_size": 16, **SMALL, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg1_s16_ls",
        vit,
        config={"patch_size": 16, **SMALL, "layer_scale_init_value": 1e-5, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg1_s16_rms_ls",
        vit,
        config={
            "patch_size": 16,
            **SMALL,
            "layer_scale_init_value": 1e-5,
            "num_reg_tokens": 1,
            "norm_layer_type": "RMSNorm",
        },
    )
    registry.register_model_config(
        "vit_reg1_s14",
        vit,
        config={"patch_size": 14, **SMALL, "num_reg_tokens": 1},
    )
    registry.register_model_config(
        "vit_reg4_m32",
        vit,
        config={"patch_size": 32, **MEDIUM, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_m16",
        vit,
        config={"patch_size": 16, **MEDIUM, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_m16_rms_avg",
        vit,
        config={"patch_size": 16, **MEDIUM, "num_reg_tokens": 4, "class_token": False, "norm_layer_type": "RMSNorm"},
    )
    registry.register_model_config(
        "vit_reg4_m14",
        vit,
        config={"patch_size": 14, **MEDIUM, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_b32",
        vit,
        config={"patch_size": 32, **BASE, "num_reg_tokens": 4, "drop_path_rate": 0.0},  # Override the BASE definition
    )
    registry.register_model_config(
        "vit_reg4_b16",
        vit,
        config={"patch_size": 16, **BASE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_b16_avg",
        vit,
        config={"patch_size": 16, **BASE, "num_reg_tokens": 4, "class_token": False},
    )
    registry.register_model_config(
        "vit_reg4_b14",
        vit,
        config={"patch_size": 14, **BASE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg8_b14_ap",
        vit,
        config={"patch_size": 14, **BASE, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_reg4_so150m_p16_avg",
        vit,
        config={"patch_size": 16, **SO150, "num_reg_tokens": 4, "class_token": False},
    )
    registry.register_model_config(
        "vit_reg8_so150m_p16_swiglu_ap",
        vit,
        config={
            "patch_size": 16,
            **SO150,
            "num_reg_tokens": 8,
            "class_token": False,
            "attn_pool_head": True,
            "mlp_layer_type": "SwiGLU_FFN",
        },
    )
    registry.register_model_config(
        "vit_reg4_so150m_p14_avg",
        vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 4, "class_token": False},
    )
    registry.register_model_config(
        "vit_reg4_so150m_p14_ls",
        vit,
        config={"patch_size": 14, **SO150, "layer_scale_init_value": 1e-5, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_so150m_p14_ap",
        vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 4, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_reg4_so150m_p14_aps",
        vit,
        config={
            "patch_size": 14,
            **SO150,
            "num_reg_tokens": 4,
            "class_token": False,
            "attn_pool_head": True,
            "attn_pool_special_tokens": True,
        },
    )
    registry.register_model_config(
        "vit_reg8_so150m_p14_avg",
        vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 8, "class_token": False},
    )
    registry.register_model_config(
        "vit_reg8_so150m_p14_swiglu",
        vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 8, "mlp_layer_type": "SwiGLU_FFN"},
    )
    registry.register_model_config(
        "vit_reg8_so150m_p14_swiglu_avg",
        vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 8, "class_token": False, "mlp_layer_type": "SwiGLU_FFN"},
    )
    registry.register_model_config(
        "vit_reg8_so150m_p14_ap",
        vit,
        config={"patch_size": 14, **SO150, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_reg4_l32",
        vit,
        config={"patch_size": 32, **LARGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_l16",
        vit,
        config={"patch_size": 16, **LARGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg8_l16_avg",
        vit,
        config={"patch_size": 16, **LARGE, "num_reg_tokens": 8, "class_token": False},
    )
    registry.register_model_config(
        "vit_reg8_l16_aps",
        vit,
        config={
            "patch_size": 16,
            **LARGE,
            "num_reg_tokens": 8,
            "class_token": False,
            "attn_pool_head": True,
            "attn_pool_special_tokens": True,
        },
    )
    registry.register_model_config(
        "vit_reg4_l14",
        vit,
        config={"patch_size": 14, **LARGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(  # DeiT III style
        "vit_reg4_l14_nps_ls",
        vit,
        config={
            "pos_embed_special_tokens": False,
            "patch_size": 14,
            **LARGE,
            "layer_scale_init_value": 1e-5,
            "num_reg_tokens": 4,
        },
    )
    registry.register_model_config(
        "vit_reg8_l14_ap",
        vit,
        config={"patch_size": 14, **LARGE, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_reg8_l14_rms_ap",
        vit,
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
        "vit_reg8_so400m_p14_ap",
        vit,
        config={"patch_size": 14, **SO400, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
    )
    registry.register_model_config(
        "vit_reg4_h16",
        vit,
        config={"patch_size": 16, **HUGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(
        "vit_reg4_h14",
        vit,
        config={"patch_size": 14, **HUGE, "num_reg_tokens": 4},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "vit_reg4_g16",
        vit,
        config={"patch_size": 16, **GIANT, "num_reg_tokens": 4},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "vit_reg4_g14",
        vit,
        config={"patch_size": 14, **GIANT, "num_reg_tokens": 4},
    )
    registry.register_model_config(  # From "Scaling Vision Transformers"
        "vit_reg4_gigantic14",
        vit,
        config={"patch_size": 14, **GIGANTIC, "num_reg_tokens": 4},
    )
