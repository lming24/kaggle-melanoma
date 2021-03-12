#!/usr/bin/env python3
"""Unit testing for datasets."""

# pylint: disable=missing-docstring, protected-access
import pathlib
import unittest

import data_loader.data_loaders as dataloaders


class TestDataLoader(unittest.TestCase):
    def test_melanoma_dataloader(self):
        example_path = pathlib.Path(__file__).parent.parent / "test_files" / "example_data"

        data = dataloaders.MelanomaDataloader(example_path,
                                              batch_size=2,
                                              shuffle=False,
                                              num_workers=0,
                                              augmentation=True)
        for item in data:
            self.assertEqual(2, item['img'].size(0))

    def test_melanoma_count_dataloader(self):
        example_path = pathlib.Path(__file__).parent.parent / "test_files" / "example_data"

        data = dataloaders.MelanomaCountDataloader(example_path,
                                                   batch_size=2,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   num_bags=6,
                                                   augmentation=True)
        for item in data:
            self.assertEqual(2, item["target"].size(0))
            self.assertEqual(2, item["len"].size(0))
            self.assertEqual(item["len"].sum().item(), item["imgs"].size(0))
