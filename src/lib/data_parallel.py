"""
Data parallel wrappers that give access to the wrapped module attributes
"""

from torch import nn
import torch


class CustomDataParallel(nn.DataParallel):
    """
    Data parallel class that gives access to the attributes of the wrapped module
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class CustomDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    Distributed data parallel class that gives access to the attributes of the wrapped module
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
