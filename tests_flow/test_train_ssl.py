import unittest

import torch

from birder.common.fs_ops import load_model
from birder.datahub.classification import TestDataset
from birder.scripts import train_vicreg


class TestSSLTrainFlow(unittest.TestCase):
    def test_train_vicreg(self) -> None:
        # Download dataset
        training_dataset = TestDataset(download=True, progress_bar=False)

        # Training
        args = train_vicreg.args_from_dict(
            network="mobilenet_v1",
            net_param=0.5,
            mlp_dim=64,
            num_workers=1,
            batch_size=2,
            drop_last=True,
            opt="lars",
            lr=0.1,
            epochs=2,
            stop_epoch=1,
            size=64,
            cpu=True,
            data_path=[training_dataset.root],
        )
        train_vicreg.train(args)

        # Load trained backbone
        device = torch.device("cpu")
        (_, model_info) = load_model(device, "convnext_v2_atto", tag="mim", epoch=1, inference=False)
        self.assertEqual(len(model_info.class_to_idx), 0)
        self.assertEqual(len(model_info.class_to_idx), model_info.signature["outputs"][0]["data_shape"][1])
