import logging
import math
import unittest

import torch

from birder import net
from birder import optim

logging.disable(logging.CRITICAL)


class TestOptimizer(unittest.TestCase):
    def test_lamb(self) -> None:
        # Just a simple sanity test
        size = (64, 64)
        n = net.MobileNet_v1(3, 10, net_param=0.5, size=size)
        optimizer = optim.Lamb(n.parameters(), lr=0.1)
        repr(optimizer)

        for _ in range(5):
            optimizer.zero_grad()
            out = n.embedding(torch.rand((4, 3, *size)))
            loss = out.abs().sum()
            loss.backward()
            optimizer.step()

            self.assertFalse(math.isnan(loss.item()))

    def test_lars(self) -> None:
        # Just a simple sanity test
        size = (64, 64)
        n = net.MobileNet_v1(3, 10, net_param=0.5, size=size)
        optimizer = optim.Lars(n.parameters(), lr=0.1)
        repr(optimizer)

        for _ in range(5):
            optimizer.zero_grad()
            out = n.embedding(torch.rand((4, 3, *size)))
            loss = out.abs().sum()
            loss.backward()
            optimizer.step()

            self.assertFalse(math.isnan(loss.item()))
