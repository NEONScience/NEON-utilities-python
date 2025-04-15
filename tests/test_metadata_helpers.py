# -*- coding: utf-8 -*-
"""
Created on Wed Apr 9 01:34:00 2025

@author: bhass

Unit tests for metadata_helpers.py
"""

import unittest
from src.neonutilities.helper_mods.metadata_helpers import convert_byte_size


class TestConvertByteSize(unittest.TestCase):
    def test_convert_byte_size(self):
        """
        Test the convert_byte_size function with various byte sizes.
        """
        # Test bytes
        self.assertEqual(convert_byte_size(500), "500.0 B")

        # Test kilobytes
        self.assertEqual(convert_byte_size(1024), "1.0 KB")
        self.assertEqual(convert_byte_size(1536), "1.5 KB")

        # Test megabytes
        self.assertEqual(convert_byte_size(1048576), "1.0 MB")
        self.assertEqual(convert_byte_size(1572864), "1.5 MB")

        # Test gigabytes
        self.assertEqual(convert_byte_size(1073741824), "1.0 GB")
        self.assertEqual(convert_byte_size(1610612736), "1.5 GB")

        # Test terabytes
        self.assertEqual(convert_byte_size(1099511627776), "1.0 TB")
        self.assertEqual(convert_byte_size(1649267441664), "1.5 TB")


if __name__ == "__main__":
    unittest.main()
