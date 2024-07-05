import logging
import unittest
from typing import Optional

import torch
from parameterized import parameterized

from birder.core.net.base import net_factory
from birder.core.net.pretraining import base

logging.disable(logging.CRITICAL)


class TestNetPretrain(unittest.TestCase):
    @parameterized.expand(  # type: ignore[misc]
        [
            ("fcmae", None, ("convnext_v2", 0)),
            ("mae_vit", None, ("vit", 0)),
            ("mae_vit", None, ("vitreg4", 1)),
            ("simmim", None, ("swin_transformer_v2", 0)),
            ("simmim", None, ("swin_transformer_v2_w2", 0)),
        ]
    )
    def test_net_pretrain(
        self, network_name: str, net_param: Optional[float], encoder_params: tuple[str, float]
    ) -> None:
        encoder = net_factory(encoder_params[0], 3, 10, encoder_params[1])
        n = base.pretrain_net_factory(network_name, encoder, net_param=net_param)
        size = n.default_size
        encoder.adjust_size(size)

        out = n(torch.rand((1, 3, size, size)))
        for key in ["loss", "pred", "mask"]:
            self.assertFalse(torch.isnan(out[key]).any())