#!/usr/bin/env python3
"""Unit testing for datasets."""

# pylint: disable=missing-docstring, protected-access
import pathlib
import unittest

import data_loader.data_sets as datasets


class TestDataSets(unittest.TestCase):
    def test_melanoma_dataset(self):
        example_path = pathlib.Path(__file__).parent.parent / "test_files" / "example_data"

        data = datasets.MelanomaDataset(example_path)
        self.assertEqual(2, len(data))
        for item in data:
            self.assertIn("patient_id", item)
            self.assertIn("sex", item)
            self.assertIn("age", item)
            self.assertIn("location", item)
            self.assertIn("target", item)
            self.assertIn("img", item)
