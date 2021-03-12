"""
Module containing the base class for a metric
"""

from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseMetric(nn.Module, metaclass=ABCMeta):
    """
    Base class for metrics. This inherits from pytorch's built-in nn.Module but
    also allows the user to give a name to the metric. This is particularly useful
    when a metric can be parameterized and multiple instances of the metric with
    different parameters are required to be calculated, since a unique name based
    on the given parameters can be given
    """
    @abstractmethod
    def forward(self, output, target):  # pylint: disable=arguments-differ
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        return self.__class__.__name__
