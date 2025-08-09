import logging
import unittest

import torch

from birder import net
from birder import scheduler

logging.disable(logging.CRITICAL)


class TestScheduler(unittest.TestCase):
    def test_cooldown(self) -> None:
        # Just a simple sanity test
        size = (64, 64)
        n = net.MobileNet_v2(3, 10, config={"alpha": 0.5}, size=size)
        optimizer = torch.optim.SGD(n.parameters(), lr=0.1)
        lr_scheduler = scheduler.CooldownLR(optimizer, total_steps=5)

        optimizer.zero_grad()
        optimizer.step()

        lr_scheduler.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        self.assertAlmostEqual(last_lr, 0.08)

        lr_scheduler.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        self.assertAlmostEqual(last_lr, 0.06)

        lr_scheduler.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        self.assertAlmostEqual(last_lr, 0.04)

        lr_scheduler.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        self.assertAlmostEqual(last_lr, 0.02)

        lr_scheduler.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        self.assertAlmostEqual(last_lr, lr_scheduler.eps)

        lr_scheduler.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        self.assertEqual(last_lr, 0.0)
