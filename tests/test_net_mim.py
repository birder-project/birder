import json
import logging
import unittest
from typing import Optional

import torch
from parameterized import parameterized

from birder.model_registry import registry
from birder.net.mim import base  # pylint: disable=unused-import # noqa: F401

logging.disable(logging.CRITICAL)


class TestNetMIM(unittest.TestCase):
    @parameterized.expand(  # type: ignore[misc]
        [
            ("crossmae", None, ("deit3_t16", None)),
            ("crossmae", None, ("deit3_reg4_t16", None)),
            ("crossmae", None, ("rope_vit_b32", None)),
            ("crossmae", None, ("rope_vitreg4_b32", None)),
            ("crossmae", None, ("rope_vit_so150m_p14_ap", None)),
            ("crossmae", None, ("rope_vitreg4_so150m_p14_ap", None)),
            ("crossmae", None, ("simple_vit_b32", None)),
            ("crossmae", None, ("vit_b32", None)),
            ("crossmae", None, ("vitreg4_b32", None)),
            ("crossmae", None, ("vit_so150m_p14_ap", None)),
            ("crossmae", None, ("vitreg4_so150m_p14_ap", None)),
            ("fcmae", None, ("convnext_v2_atto", None)),
            ("fcmae", None, ("regnet_y_200m", None)),
            ("fcmae", None, ("regnet_z_500m", None)),
            ("mae_hiera", None, ("hiera_tiny", None)),
            ("mae_hiera", None, ("hiera_abswin_tiny", None)),
            ("mae_vit", None, ("deit3_t16", None)),
            ("mae_vit", None, ("deit3_reg4_t16", None)),
            ("mae_vit", None, ("rope_vit_b32", None)),
            ("mae_vit", None, ("rope_vitreg4_b32", None)),
            ("mae_vit", None, ("rope_vit_so150m_p14_ap", None)),
            ("mae_vit", None, ("rope_vitreg4_so150m_p14_ap", None)),
            ("mae_vit", None, ("simple_vit_b32", None)),
            ("mae_vit", None, ("vit_b32", None)),
            ("mae_vit", None, ("vitreg4_b32", None)),
            ("mae_vit", None, ("vit_so150m_p14_ap", None)),
            ("mae_vit", None, ("vitreg4_so150m_p14_ap", None)),
            ("simmim", None, ("maxvit_t", None)),
            ("simmim", None, ("nextvit_s", None)),
            ("simmim", None, ("regnet_y_800m", None)),
            ("simmim", None, ("swin_transformer_v2_t", None)),
            ("simmim", None, ("swin_transformer_v2_w2_t", None)),
            ("simmim", None, ("vit_b32", None)),
            ("simmim", None, ("vitreg4_b32", None)),
            ("simmim", None, ("vit_so150m_p14_ap", None)),
            ("simmim", None, ("vitreg4_so150m_p14_ap", None)),
        ]
    )
    def test_net_mim(self, network_name: str, net_param: Optional[float], encoder_params: tuple[str, float]) -> None:
        encoder = registry.net_factory(encoder_params[0], 3, 10, net_param=encoder_params[1])
        size = (encoder.max_stride * 6, encoder.max_stride * 6)
        encoder.adjust_size(size)
        n = registry.mim_net_factory(network_name, encoder, net_param=net_param, size=size)

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        out = n(torch.rand((1, 3, *size)))
        for key in ["loss", "pred", "mask"]:
            self.assertFalse(torch.isnan(out[key]).any())

        self.assertEqual(out["loss"].ndim, 0)
