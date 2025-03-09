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
from birder.tools.avg_model import avg_models


class TestTrainEvaluateFlow(unittest.TestCase):
    def test_train_pretrained(self) -> None:
        model_list = birder.list_pretrained_models("mobilenet_v3_small*")
        model_name = random.choice(model_list)

        # Download model
        birder.load_pretrained_model(model_name, progress_bar=False)

        # Download dataset
        training_dataset = TestDataset(download=True, progress_bar=False)
        validation_dataset = TestDataset(split="validation")

        # Linear probing
        model_metadata = registry.get_pretrained_metadata(model_name)
        network = model_metadata["net"]["network"]
        net_param = model_metadata["net"].get("net_param", None)
        tag = model_metadata["net"].get("tag", None)
        args = train.args_from_dict(
            network=network,
            net_param=net_param,
            tag=tag,
            pretrained=True,
            freeze_body=True,
            num_workers=1,
            batch_size=2,
            drop_last=True,
            lr=0.1,
            epochs=2,
            size=64,
            cpu=True,
            save_frequency=1,
            data_path=training_dataset.root,
            val_path=validation_dataset.root,
        )
        train.train(args)

        # Check checkpoints are valid
        device = torch.device("cpu")
        (net, (class_to_idx, signature, rgb_stats, *_)) = load_model(
            device, network, net_param=net_param, tag=tag, epoch=1, inference=True
        )
        self.assertEqual(len(class_to_idx), signature["outputs"][0]["data_shape"][1])
        (net, (class_to_idx, signature, rgb_stats, *_)) = load_model(
            device, network, net_param=net_param, tag=tag, epoch=2, inference=True
        )
        self.assertEqual(len(class_to_idx), signature["outputs"][0]["data_shape"][1])

        # Average checkpoints
        avg_models(network, net_param, tag, reparameterized=False, epochs=[1, 2], force=True)

        # Check average checkpoint is valid
        device = torch.device("cpu")
        (net, (class_to_idx, signature, rgb_stats, *_)) = load_model(
            device, network, net_param=net_param, tag=tag, epoch=0, inference=True
        )

        # Check valid config
        network_name = get_network_name(network, net_param, tag)
        config = read_config(network_name)
        weights_path = model_path(network_name, epoch=0)
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
