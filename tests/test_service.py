import logging
import unittest

from torchvision.transforms import v2

from birder.core.transforms.classification import get_rgb_values
from birder.core.transforms.classification import inference_preset

logging.disable(logging.CRITICAL)
from birder.service import classification  # noqa: E402 # pylint: disable=wrong-import-position


class TestService(unittest.TestCase):
    def test_classification(self) -> None:
        service_preset: v2.Compose = classification.inference_preset(224, 1.0, get_rgb_values("none"))  # type:ignore
        core_preset: v2.Compose = inference_preset(224, 1.0, get_rgb_values("none"))

        self.assertIsInstance(service_preset, v2.Compose)
        self.assertIsInstance(core_preset, v2.Compose)
        for service_transform, core_transform in zip(service_preset.transforms, core_preset.transforms):
            self.assertEqual(type(service_transform), type(core_transform))