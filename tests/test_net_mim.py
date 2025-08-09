import json
import logging
import unittest

import torch
from parameterized import parameterized

from birder.model_registry import registry
from birder.net.mim import base  # pylint: disable=unused-import # noqa: F401

logging.disable(logging.CRITICAL)


class TestNetMIM(unittest.TestCase):
    @parameterized.expand(  # type: ignore[misc]
        [
            ("crossmae", "deit3_t16"),
            ("crossmae", "deit3_reg4_t16"),
            ("crossmae", "rope_vit_b32"),
            ("crossmae", "rope_vit_reg4_b32"),
            ("crossmae", "rope_vit_so150m_p14_ap"),
            ("crossmae", "rope_vit_reg4_so150m_p14_ap"),
            ("crossmae", "simple_vit_b32"),
            ("crossmae", "vit_b32"),
            ("crossmae", "vit_reg4_b32"),
            ("crossmae", "vit_so150m_p14_ap"),
            ("crossmae", "vit_reg4_so150m_p14_ap"),
            ("crossmae", "vit_parallel_s16_18x2_ls"),
            ("fcmae", "convnext_v2_atto"),
            ("fcmae", "regnet_y_200m"),
            ("fcmae", "regnet_z_500m"),
            ("mae_hiera", "hiera_tiny"),
            ("mae_hiera", "hiera_abswin_tiny"),
            ("mae_vit", "deit3_t16"),
            ("mae_vit", "deit3_reg4_t16"),
            ("mae_vit", "rope_vit_b32"),
            ("mae_vit", "rope_vit_reg4_b32"),
            ("mae_vit", "rope_vit_so150m_p14_ap"),
            ("mae_vit", "rope_vit_reg4_so150m_p14_ap"),
            ("mae_vit", "simple_vit_b32"),
            ("mae_vit", "vit_b32"),
            ("mae_vit", "vit_reg4_b32"),
            ("mae_vit", "vit_so150m_p14_ap"),
            ("mae_vit", "vit_reg4_so150m_p14_ap"),
            ("mae_vit", "vit_parallel_s16_18x2_ls"),
            ("simmim", "hieradet_tiny"),
            ("simmim", "maxvit_t"),
            ("simmim", "nextvit_s"),
            ("simmim", "regnet_y_800m"),
            ("simmim", "swin_transformer_v2_t"),
            ("simmim", "swin_transformer_v2_w2_t"),
            ("simmim", "vit_b32"),
            ("simmim", "vit_reg4_b32"),
            ("simmim", "vit_so150m_p14_ap"),
            ("simmim", "vit_reg4_so150m_p14_ap"),
            ("simmim", "vit_parallel_s16_18x2_ls"),
        ]
    )
    def test_net_mim(self, network_name: str, encoder: str) -> None:
        encoder = registry.net_factory(encoder, 3, 10)
        size = (encoder.max_stride * 6, encoder.max_stride * 6)
        encoder.adjust_size(size)
        n = registry.mim_net_factory(network_name, encoder, size=size)

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        out = n(torch.rand((1, 3, *size)))
        for key in ["loss", "pred", "mask"]:
            self.assertFalse(torch.isnan(out[key]).any())

        self.assertEqual(out["loss"].ndim, 0)
