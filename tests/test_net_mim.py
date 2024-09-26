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
            ("fcmae", None, ("convnext_v2_atto", 0)),
            ("mae_vit", None, ("simple_vit_b32", 0)),
            ("mae_vit", None, ("vit_b32", 0)),
            ("mae_vit", None, ("vitreg4_b32", 1)),
            ("simmim", None, ("maxvit_t", 0)),
            ("simmim", None, ("nextvit_s", 0)),
            ("simmim", None, ("swin_transformer_v2_t", 0)),
            ("simmim", None, ("swin_transformer_v2_w2_t", 0)),
        ]
    )
    def test_net_mim(self, network_name: str, net_param: Optional[float], encoder_params: tuple[str, float]) -> None:
        encoder = registry.net_factory(encoder_params[0], 3, 10, net_param=encoder_params[1])
        n = registry.mim_net_factory(network_name, encoder, net_param=net_param)
        size = n.default_size
        encoder.adjust_size(size)

        out = n(torch.rand((1, 3, size, size)))
        for key in ["loss", "pred", "mask"]:
            self.assertFalse(torch.isnan(out[key]).any())
