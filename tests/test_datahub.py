import logging
import unittest
from unittest.mock import mock_open
from unittest.mock import patch

from birder.core.datahub import _lib

logging.disable(logging.CRITICAL)


class TestDataHub(unittest.TestCase):
    def test_lib(self) -> None:
        m = mock_open(read_data=b"test data")
        with patch("builtins.open", m):
            hex_digest = _lib.calc_sha256("some_file.tar.gz")
            m.assert_called_with("some_file.tar.gz", "rb")
            self.assertEqual(hex_digest, "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9")
