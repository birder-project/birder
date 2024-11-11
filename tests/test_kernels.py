import logging
import unittest

from birder.kernels import load_kernel

logging.disable(logging.CRITICAL)


class TestKernels(unittest.TestCase):
    def test_deformable_detr(self) -> None:
        load_kernel.load_msda()
