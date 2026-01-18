import unittest

import torch

from birder.common.fs_ops import load_mim_checkpoint
from birder.common.fs_ops import load_model
from birder.datahub.classification import TestDataset
from birder.scripts import train_mim


class TestTrainMIMFlow(unittest.TestCase):
    def test_train_mim(self) -> None:
        # Download dataset
        training_dataset = TestDataset(download=True, progress_bar=False)

        # Training
        args = train_mim.args_from_dict(
            network="fcmae",
            encoder="convnext_v2_atto",
            encoder_model_config={"drop_path_rate": 0.0},
            num_workers=1,
            batch_size=2,
            drop_last=True,
            lr=0.1,
            epochs=2,
            stop_epoch=1,
            size=64,
            cpu=True,
            data_path=[training_dataset.root],
        )
        train_mim.train(args)

        # Resume training
        args = train_mim.args_from_dict(
            network="fcmae",
            encoder="convnext_v2_atto",
            encoder_model_config={"drop_path_rate": 0.0},
            num_workers=1,
            batch_size=2,
            drop_last=True,
            lr=0.1,
            epochs=2,
            resume_epoch=1,
            load_states=True,
            size=64,
            cpu=True,
            data_path=[training_dataset.root],
        )
        train_mim.train(args)

        # Load trained model
        device = torch.device("cpu")
        _ = load_mim_checkpoint(device, "fcmae", encoder="convnext_v2_atto", epoch=1)

        # Load trained encoder
        _, model_info = load_model(device, "convnext_v2_atto", tag="mim", epoch=1, inference=False)
        self.assertEqual(len(model_info.class_to_idx), 0)
        self.assertEqual(len(model_info.class_to_idx), model_info.signature["outputs"][0]["data_shape"][1])
