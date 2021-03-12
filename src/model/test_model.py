#!/usr/bin/env python3
"""Unit testing for models."""

# pylint: disable=missing-docstring, protected-access
import pathlib
import unittest

import torch
import model.model as models
import data_loader.data_loaders as dataloaders


class TestModels(unittest.TestCase):
    def test_resnet_model(self):
        example_path = pathlib.Path(__file__).parent.parent / "test_files" / "example_data"
        data = dataloaders.MelanomaDataloader(example_path, batch_size=2, shuffle=False, num_workers=0)

        model = models.ResNet('18', 2, additional_features=8, pretrained=False)
        with torch.no_grad():
            for item in data:
                result = model(item)
                self.assertEqual(2, result.size(0))
                self.assertEqual(2, result.size(1))

    def test_efficientnet_model(self):
        example_path = pathlib.Path(__file__).parent.parent / "test_files" / "example_data"
        data = dataloaders.MelanomaDataloader(example_path, batch_size=2, shuffle=False, num_workers=0)

        model = models.EfficientNetwork('b0', 2, additional_features=8)
        with torch.no_grad():
            for item in data:
                result = model(item)
                self.assertEqual(2, result.size(0))
                self.assertEqual(2, result.size(1))
