import logging
import os
import unittest
import warnings

import requests

from birder.conf import settings
from birder.core.net.base import BaseNet
from birder.core.net.detection.base import DetectionBaseNet
from birder.core.net.pretraining.base import PreTrainBaseNet
from birder.model_registry import registry
from birder.model_registry.model_registry import ModelRegistry

logging.disable(logging.CRITICAL)


class TestRegistry(unittest.TestCase):
    def test_registry_nets(self) -> None:
        for net in registry.nets.values():
            self.assertTrue(issubclass(net, BaseNet))

        for net in registry.detection_nets.values():
            self.assertTrue(issubclass(net, DetectionBaseNet))

        for net in registry.pretrain_nets.values():
            self.assertTrue(issubclass(net, PreTrainBaseNet))

    def test_no_duplicates(self) -> None:
        all_names = []
        for net_name in registry.nets:
            all_names.append(net_name)

        for net_name in registry.detection_nets:
            all_names.append(net_name)

        for net_name in registry.pretrain_nets:
            all_names.append(net_name)

        self.assertEqual(len(all_names), len(set(all_names)))

    @unittest.skipUnless(os.environ.get("NETWORK_TESTS", False), "Avoid tests that requiring network access")
    def test_manifest(self) -> None:
        for model_name, model_info in registry.pretrained_nets.items():
            for model_format in model_info["formats"]:
                url = f"{settings.REGISTRY_BASE_UTL}/{model_name}.{model_format}"
                resp = requests.head(url, timeout=5, allow_redirects=True)
                self.assertEqual(resp.status_code, 200)
                self.assertGreater(int(resp.headers["Content-Length"]), 100000)


class TestModelRegistry(unittest.TestCase):
    def test_model_registry(self) -> None:
        model_registry = ModelRegistry()
        model_registry.register_model("net1", BaseNet)
        model_registry.register_model("net2", BaseNet)
        model_registry.register_model("net3", DetectionBaseNet)
        model_registry.register_model("net4", PreTrainBaseNet)

        self.assertListEqual(list(model_registry.all_nets.keys()), ["net1", "net2", "net3", "net4"])
        self.assertListEqual(list(model_registry.nets.keys()), ["net1", "net2"])
        self.assertListEqual(list(model_registry.detection_nets.keys()), ["net3"])
        self.assertListEqual(list(model_registry.pretrain_nets.keys()), ["net4"])

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            model_registry.register_model("net1", BaseNet)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))