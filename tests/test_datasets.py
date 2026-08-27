import logging
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from birder.data.collators import naflex as naflex_collators
from birder.data.datasets import coco
from birder.data.datasets import directory
from birder.data.datasets import fake
from birder.data.datasets import naflex as naflex_datasets
from birder.data.datasets import webdataset

logging.disable(logging.CRITICAL)


class TestDatasets(unittest.TestCase):
    def test_naflex_multiscale_dataset_transforms_during_loading(self) -> None:
        events: list[tuple[str, int]] = []

        class TestDataset(torch.utils.data.Dataset):
            def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
                events.append(("load", index))
                return (torch.full((1, 4, 4), index), index)

            def __len__(self) -> int:
                return 2

        def transform(image: torch.Tensor) -> torch.Tensor:
            index = int(image[0, 0, 0].item())
            events.append(("transform", index))
            return image[:, :2, :2]

        spec = naflex_collators.NaFlexBatchSpec(2, 1)
        batch_processor = naflex_collators.NaFlexBatchProcessor(
            naflex_collators.NaFlexTrainingCollator(),
            (spec,),
            {spec: transform},
            seed=0,
        )
        dataset = naflex_datasets.NaFlexMultiScaleDataset(TestDataset(), batch_processor)
        loader = DataLoader(dataset, batch_size=2, collate_fn=batch_processor.base_collator)

        (patches, grid_sizes, valid_mask), targets = next(iter(loader))

        self.assertSequenceEqual(events, [("load", 0), ("transform", 0), ("load", 1), ("transform", 1)])
        self.assertEqual(patches.size(), (2, 1, 4))
        torch.testing.assert_close(grid_sizes, torch.ones((2, 2), dtype=torch.int64))
        self.assertTrue(valid_mask.all().item())
        torch.testing.assert_close(targets, torch.tensor([0, 1]))

    def test_fake_data_with_paths(self) -> None:
        dataset = fake.FakeDataWithPaths(
            size=4,
            image_size=(3, 8, 8),
            num_classes=2,
            transform=lambda _: torch.zeros((3, 8, 8)),
        )

        self.assertEqual(len(dataset), 4)
        path, sample, label = dataset[2]
        self.assertEqual(path, "fake/path/2.jpeg")
        self.assertTrue(torch.equal(sample, torch.zeros((3, 8, 8))))
        self.assertIn(label, (0, 1))

    def test_directory(self) -> None:
        dataset = directory.ImageListDataset(
            [("file1.jpeg", 0), ("file2.jpeg", 1), ("file3.jpeg", 0), ("file4.jpeg", 0)],
            transforms=lambda x: x + ".data",
            loader=lambda x: x,
        )

        self.assertEqual(len(dataset), 4)
        path, sample, label = dataset[2]
        self.assertEqual(path, "file3.jpeg")
        self.assertEqual(sample, "file3.jpeg.data")
        self.assertEqual(label, 0)

        repr(dataset)

    def test_webdataset(self) -> None:
        sample_name, data, label = webdataset.decode_sample_name(("shard1", "sample6", b"data", 1))
        self.assertEqual(sample_name, "shard1/sample6")
        self.assertEqual(data, b"data")
        self.assertEqual(label, 1)

    def test_wds_args_from_multiple_info(self) -> None:
        with patch(
            "birder.data.datasets.webdataset.fs_ops.read_wds_info",
            side_effect=[
                {"splits": {"training": {"num_samples": 10, "filenames": ["train-000000.tar", "train-000001.tar"]}}},
                {"splits": {"training": {"num_samples": 5, "filenames": ["train-000000.tar"]}}},
            ],
        ):
            filenames, size = webdataset.wds_args_from_info(
                ["/datasets/part1/_info.json", "/datasets/part2/_info.json"], "training"
            )

        self.assertEqual(
            filenames,
            [
                "/datasets/part1/train-000000.tar",
                "/datasets/part1/train-000001.tar",
                "/datasets/part2/train-000000.tar",
            ],
        )
        self.assertEqual(size, 15)

    def test_wds_args_from_multiple_info_with_different_splits(self) -> None:
        with patch(
            "birder.data.datasets.webdataset.fs_ops.read_wds_info",
            side_effect=[
                {"splits": {"train": {"num_samples": 10, "filenames": ["train-000000.tar"]}}},
                {"splits": {"training": {"num_samples": 5, "filenames": ["training-000000.tar"]}}},
            ],
        ):
            filenames, size = webdataset.wds_args_from_info(
                ["/datasets/part1/_info.json", "/datasets/part2/_info.json"], ["train", "training"]
            )

        self.assertEqual(
            filenames,
            ["/datasets/part1/train-000000.tar", "/datasets/part2/training-000000.tar"],
        )
        self.assertEqual(size, 15)

    def test_wds_args_from_remote_info(self) -> None:
        with patch(
            "birder.data.datasets.webdataset.fs_ops.read_wds_info",
            return_value={
                "splits": {
                    "validation": {
                        "num_samples": 2,
                        "filenames": ["validation-000000.tar", "https://cdn.example.com/custom.tar"],
                    }
                }
            },
        ):
            filenames, size = webdataset.wds_args_from_info(
                "https://huggingface.co/datasets/birder/example/resolve/main/_info.json", "validation"
            )

        self.assertEqual(
            filenames,
            [
                "https://huggingface.co/datasets/birder/example/resolve/main/validation-000000.tar",
                "https://cdn.example.com/custom.tar",
            ],
        )
        self.assertEqual(size, 2)

    def test_wds_args_from_non_http_remote_info(self) -> None:
        with patch(
            "birder.data.datasets.webdataset.fs_ops.read_wds_info",
            return_value={"splits": {"validation": {"num_samples": 2, "filenames": ["validation-000000.tar"]}}},
        ):
            filenames, size = webdataset.wds_args_from_info("s3://bucket/datasets/example/_info.json", "validation")

        self.assertEqual(filenames, ["s3://bucket/datasets/example/validation-000000.tar"])
        self.assertEqual(size, 2)

    def test_wds_args_from_mixed_local_and_remote_info(self) -> None:
        with patch(
            "birder.data.datasets.webdataset.fs_ops.read_wds_info",
            side_effect=[
                {"splits": {"validation": {"num_samples": 3, "filenames": ["val-000000.tar"]}}},
                {"splits": {"validation": {"num_samples": 4, "filenames": ["val-000001.tar"]}}},
            ],
        ):
            filenames, size = webdataset.wds_args_from_info(
                [
                    "/datasets/local/_info.json",
                    "https://huggingface.co/datasets/birder/example/resolve/main/_info.json",
                ],
                "validation",
            )

        self.assertEqual(
            filenames,
            [
                "/datasets/local/val-000000.tar",
                "https://huggingface.co/datasets/birder/example/resolve/main/val-000001.tar",
            ],
        )
        self.assertEqual(size, 7)

    def test_decode_detection_target(self) -> None:
        image = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
        target = {
            "image_id": 42,
            "boxes": [[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 250.0]],
            "labels": [3, 7],
        }

        result_image, result_target = webdataset.decode_detection_target((image, target))

        self.assertTrue(torch.equal(result_image, image))
        self.assertEqual(result_target["image_id"], 42)
        self.assertEqual(result_target["boxes"].shape, (2, 4))
        self.assertEqual(result_target["labels"].tolist(), [3, 7])
        self.assertEqual(result_target["labels"].dtype, torch.int64)

    def test_decode_detection_target_label_remap(self) -> None:
        image = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
        target = {
            "image_id": 42,
            "boxes": [[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 250.0]],
            "labels": [3, 7],
        }
        label_remap = {3: 1, 7: 1}

        _, result_target = webdataset.decode_detection_target((image, target), label_remap=label_remap)

        self.assertEqual(result_target["labels"].tolist(), [1, 1])

    def test_decode_detection_target_empty(self) -> None:
        image = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
        target = {
            "image_id": 99,
            "boxes": [],
            "labels": [],
        }

        _, result_target = webdataset.decode_detection_target((image, target))

        self.assertEqual(result_target["image_id"], 99)
        self.assertEqual(result_target["boxes"].shape, (0, 4))
        self.assertEqual(result_target["labels"].shape, (0,))

    def test_decode_detection_sample_with_names_and_orig_size(self) -> None:
        image = torch.randint(0, 255, (3, 8, 12), dtype=torch.uint8)
        target = {
            "image_id": 7,
            "boxes": [[1.0, 2.0, 6.0, 7.0]],
            "labels": [3],
        }

        sample_name, _, result_target, orig_size = webdataset.decode_detection_sample(
            ("dataset.tar", "sample000007", image, target)
        )

        self.assertEqual(sample_name, "dataset.tar/sample000007")
        self.assertEqual(result_target["image_id"], 7)
        self.assertEqual(tuple(orig_size), (8, 12))
        self.assertEqual(result_target["labels"].tolist(), [3])

    def test_coco_mapped_class_to_idx(self) -> None:
        class_to_idx = {"class-b": 10, "class-a": 2, "class-c": 40}
        label_mapping = {"class-a": "family-1", "class-b": "family-2", "class-c": "family-2"}

        mapped_class_to_idx = coco._mapped_class_to_idx(class_to_idx, label_mapping)
        self.assertEqual(mapped_class_to_idx, {"family-1": 1, "family-2": 2})

    def test_coco_mapped_class_to_idx_missing_mapping(self) -> None:
        class_to_idx = {"class-a": 1, "class-b": 2}
        label_mapping = {"class-a": "family-1"}

        with self.assertRaises(ValueError):
            coco._mapped_class_to_idx(class_to_idx, label_mapping)

    def test_build_label_mapping_indices(self) -> None:
        class_to_idx = {"class-b": 10, "class-a": 2, "class-c": 40}
        label_mapping = {"class-a": "family-1", "class-b": "family-2", "class-c": "family-2"}

        mapped_class_to_idx, label_remap = coco.build_label_mapping_indices(class_to_idx, label_mapping)

        self.assertEqual(mapped_class_to_idx, {"family-1": 1, "family-2": 2})
        self.assertEqual(label_remap, {2: 1, 10: 2, 40: 2})

    def test_build_coco_category_remap_zero_based_ids(self) -> None:
        categories = {
            0: {"id": 0, "name": "Deer"},
            1: {"id": 1, "name": "Hog"},
        }

        class_to_idx, label_remap = coco.build_coco_category_remap(categories)
        self.assertEqual(class_to_idx, {"Deer": 1, "Hog": 2})
        self.assertEqual(label_remap, {0: 1, 1: 2})
