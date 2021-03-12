"""
Useful utilities for distributed settings
"""

import torch.cuda
import torch.distributed as dist


def get_global_rank(default_value=0):
    """
    Returns global rank of the calling process in a distributed setting.
    The master process always has rank 0. Returns default_value=0 if not in a
    distributed environment.
    """
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return default_value


def get_local_rank(default_value=0):
    """
    Returns the local rank of the calling process in distributed setting.
    Returns default_value=0 if not in a distributed environment.
    """
    return get_global_rank(default_value=default_value) % get_local_size


def get_world_size():
    """
    Returns the total number of processes running in the global process group
    """
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_local_size():
    """
    Returns the total number of processes running locally.

    Assumes that if a process group is initialized, then there will be one
    process per GPU (recommended configuration). If there are no GPUs or if
    the process group is not initialized, the local size will be 1.
    """
    if dist.is_initialized():
        # Assume one process per GPU if GPUs are found
        return max(1, torch.cuda.device_count())
    return 1
