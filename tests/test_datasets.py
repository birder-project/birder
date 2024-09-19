import logging
import unittest

from birder.datasets import directory
from birder.datasets import webdataset

logging.disable(logging.CRITICAL)


class TestDatasets(unittest.TestCase):
    def test_directory(self) -> None:
        dataset = directory.ImageListDataset(
            [("file1.jpeg", 0), ("file2.jpeg", 1), ("file3.jpeg", 0), ("file4.jpeg", 0)],
            transforms=lambda x: x + ".data",
            loader=lambda x: x,
        )

        self.assertEqual(len(dataset), 4)
        (path, sample, label) = dataset[2]
        self.assertEqual(path, "file3.jpeg")
        self.assertEqual(sample, "file3.jpeg.data")
        self.assertEqual(label, 0)

        repr(dataset)

    def test_webdataset(self) -> None:
        (sample_name, data, label) = webdataset.decode_sample_name(("shard1", "sample6", b"data", 1))
        self.assertEqual(sample_name, "shard1/sample6")
        self.assertEqual(data, b"data")
        self.assertEqual(label, 1)
