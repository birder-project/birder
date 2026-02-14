import logging
import unittest

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
