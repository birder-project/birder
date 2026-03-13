import logging
import unittest
from unittest.mock import patch

from birder.data.datasets import coco
from birder.data.datasets import directory
from birder.data.datasets import webdataset

logging.disable(logging.CRITICAL)


# pylint: disable=protected-access
class TestDatasets(unittest.TestCase):
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

    def test_coco_mapped_class_to_idx(self) -> None:
        class_to_idx = {"class-b": 10, "class-a": 2, "class-c": 40}
        label_mapping = {"class-a": "family-1", "class-b": "family-2", "class-c": "family-2"}

        mapped_class_to_idx = coco._mapped_class_to_idx(class_to_idx, label_mapping)
        self.assertEqual(mapped_class_to_idx, {"family-1": 1, "family-2": 2})

    def test_coco_mapped_class_to_idx_missing_mapping(self) -> None:
        class_to_idx = {"class-a": 1, "class-b": 2}
        label_mapping = {"class-a": "family-1"}

        with self.assertRaisesRegex(ValueError, "Missing label mapping for class 'class-b'"):
            coco._mapped_class_to_idx(class_to_idx, label_mapping)
