import logging
import unittest

import torch

from birder.model_registry import registry
from birder.net.ssl import DINO_v1
from birder.net.ssl import VICReg
from birder.net.ssl.dino_v1 import DINOHead

logging.disable(logging.CRITICAL)


class TestNetSSL(unittest.TestCase):
    def test_dino_v1(self) -> None:
        batch_size = 4
        backbone = registry.net_factory("resnet_v2_18", 3, 0)
        dino_head = DINOHead(
            backbone.embedding_size,
            128,
            use_bn=True,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=512,
            bottleneck_dim=256,
        )
        net = DINO_v1(3, backbone, dino_head)

        # Test network
        out = net(
            [
                torch.rand((batch_size, 3, 128, 128)),
                torch.rand((batch_size, 3, 128, 128)),
                torch.rand((batch_size, 3, 96, 96)),
                torch.rand((batch_size, 3, 96, 96)),
                torch.rand((batch_size, 3, 96, 96)),
                torch.rand((batch_size, 3, 96, 96)),
            ]
        )
        self.assertFalse(torch.isnan(out).any())

    def test_vicreg(self) -> None:
        batch_size = 4
        backbone = registry.net_factory("resnet_v1_18", 3, 0)
        net = VICReg(3, backbone, mlp_dim=128, batch_size=batch_size, sim_coeff=0.1, std_coeff=0.1, cov_coeff=0.1)

        # Test network
        out = net(torch.rand((batch_size, 3, 128, 128)), torch.rand((batch_size, 3, 128, 128)))
        self.assertFalse(torch.isnan(out).any())
