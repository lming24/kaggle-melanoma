"""
Various utility functions or classes used by the pipeline
"""

import itertools
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch


def read_json(fname):
    """
    Loads a json file to an OrderedDict so that the order in the file is preserved.
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """
    Writes the contents to a json file maintaining the given order of keys
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """
    Wrapper function for endless data loader.

    Useful for iteration-based training (as opposed to epoch-based training)
    """
    for loader in itertools.repeat(data_loader):
        yield from loader


def collect_tensors(input_obj):
    """
    Collects all tensors in an arbitrarily nested object. Note that
    only lists, dicts, tuples, and sets are supported.
    """
    if torch.is_tensor(input_obj):
        return [input_obj]

    if not isinstance(input_obj, (list, dict, tuple, set)):
        return []

    if isinstance(input_obj, dict):
        input_obj = input_obj.items()

    return itertools.chain(*map(collect_tensors, input_obj))


def all_tensors_to(input_obj, *args, **kwargs):
    """
    Recursively iterates an arbitrarily nested object and calls the .to() method
    on all tensors it finds while maintaining the structure of the original input object
    e.g For an object obj = [torch.tensor(1.0), {'test':[5, torch.tensor(3.0)]}]
    all_tensors_to(obj, device='cuda') will return
    [torch.tensor(1.0, device='cuda'), {'test':[5, torch.tensor(3.0, device='cuda')]}]
    """
    try:
        return input_obj.to(*args, **kwargs)
    except AttributeError:
        # Not a tensor
        pass

    if not isinstance(input_obj, (list, tuple, dict, set)):
        return input_obj

    original_input = input_obj
    if isinstance(input_obj, dict):
        input_obj = input_obj.items()

    res = []
    for i in input_obj:
        try:
            res.append(all_tensors_to(i, *args, **kwargs))
        except TypeError:
            pass
    if isinstance(original_input, dict):
        res = dict(res)
    elif isinstance(original_input, tuple):
        res = tuple(res)
    elif isinstance(original_input, set):
        res = set(res)
    return res


class MetricTracker:
    """
    Keeps track of multiple metrics and provides methods to update or
    calculate averages over time
    """
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """
        Reset all metrics to 0
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, batch_size=1):
        """
        Updates the metric indexed by `key`

        :param key: Key/name of the metric
        :param value: Value of the metric (an average of some metric over
            a mini batch)
        :param batch_size: The number of samples on which this value was calculated on
            (the batch size). Note: This will only make a difference when the batch size
            is not constant throughout each step/iteration in an epoch. In cases where
            the batch_size is constant, setting it to 1 will produce the same results
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * batch_size
        self._data.counts[key] += batch_size
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        Returns the average of a metric over time

        Returns the average of a metric since last reset. Usually a reset is called
        at the beginning of each epoch
        """
        return self._data.average[key]

    def result(self):
        """
        Returns a dictionary of the averages of all tracked metrics
        """
        return dict(self._data.average)
