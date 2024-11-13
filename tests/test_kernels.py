import logging
import unittest

import torch

from birder.kernels import load_kernel

logging.disable(logging.CRITICAL)


class TestKernels(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_deformable_detr(self) -> None:
        msda = load_kernel.load_msda()
        self.assertIsNotNone(msda)
