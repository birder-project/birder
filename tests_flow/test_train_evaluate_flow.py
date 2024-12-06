import random
import unittest

import torch
from torch.utils.data import DataLoader

import birder
from birder.common.fs_ops import load_model
from birder.common.fs_ops import model_path
from birder.common.fs_ops import read_config
from birder.common.lib import get_network_name
from birder.datahub.classification import TestDataset
from birder.model_registry import registry
from birder.scripts import train


class TestTrainFlow(unittest.TestCase):
    def test_train_pretrained(self) -> None:
        model_list = birder.list_pretrained_models("*il-common")
        model_name = random.choice(model_list)

        # Download model
        birder.load_pretrained_model(model_name, progress_bar=False)

        # Download dataset
        TestDataset(download=True, progress_bar=False)

        # Linear probing
        model_info = registry.get_pretrained_info(model_name)
        network = model_info["net"]["network"]
        net_param = net_param = model_info["net"].get("net_param", None)
        tag = model_info["net"].get("tag", None)
        args = train.args_from_dict(
            network=network,
            net_param=net_param,
            tag=tag,
            pretrained=True,
            freeze_body=True,
            num_workers=1,
            batch_size=1,
            lr=0.1,
            epochs=1,
            size=64,
            cpu=True,
            data_path="data/TestDataset/training",
            val_path="data/TestDataset/validation",
        )
        train.train(args)

        # Check checkpoint is valid
        device = torch.device("cpu")
        (net, class_to_idx, signature, rgb_stats) = load_model(
            device, network, net_param=net_param, tag=tag, epoch=1, inference=True
        )
        self.assertEqual(len(class_to_idx), signature["outputs"][0]["data_shape"][1])

        # Check valid config
        network_name = get_network_name(network, net_param, tag)
        config = read_config(network_name)
        weights_path = model_path(network_name, epoch=1)
        _ = birder.load_model_with_cfg(config, weights_path)

        # Prepare dataloader
        size = birder.get_size_from_signature(signature)
        transform = birder.classification_transform(size, rgb_stats)
        dataset = TestDataset(download=True, split="validation", transform=transform, progress_bar=False)
        inference_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
        )

        # Run evaluation
        results = birder.evaluate_classification(device, net, inference_loader, class_to_idx)

        # Ensure valid results object
        self.assertFalse(results.missing_labels)
