import logging
import unittest

import torch

from birder.model_registry import registry
from birder.net.ssl import VICReg

logging.disable(logging.CRITICAL)


class TestNetSSL(unittest.TestCase):
    def test_vicreg(self) -> None:
        batch_size = 4
        backbone = registry.net_factory("resnet_v1_18", 3, 0)
        net = VICReg(backbone, mlp_dim=128, batch_size=batch_size, sim_coeff=0.1, std_coeff=0.1, cov_coeff=0.1)

        # Test network
        out = net(torch.rand((batch_size, 3, 128, 128)), torch.rand((batch_size, 3, 128, 128)))
        self.assertFalse(torch.isnan(out).any())
