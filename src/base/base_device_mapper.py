"""
Base class for device mapper. A device mapper detects the environment the model
is running on, prepares the available devices and parallelizes models accordingly
"""

from abc import ABC, abstractmethod

import torch

import logger.logger as logger
from lib.distributed import get_global_rank
from lib.utils import all_tensors_to


class BaseDeviceMapper(ABC):
    """
    Base class for a device mapper
    """
    def __init__(self, n_gpu, auto_parallel=True):
        self.logger = logger.get_logger(self.__class__.__name__, verbosity=2)
        self.n_gpu = n_gpu

        self.logger.info("Initializing devices..")
        device, gpu_ids, n_gpu, n_processes = self.prepare_device()

        self.device = torch.device(device)
        self.gpu_ids = [torch.device(d) for d in gpu_ids]
        self.n_gpu = int(n_gpu)
        self.n_processes = int(n_processes)
        self.auto_parallel = auto_parallel

        if get_global_rank() == 0:
            self.logger.info("Number of running processes: %s", self.n_processes)
            self.logger.info("Number of usable GPUs: %s", self.n_gpu)

    @abstractmethod
    def prepare_device(self):
        """
        Prepares all devices and performs any required initial 'handshakes'.
        e.g in a distributed settings, this is where the process group would
        be initialized

        This method should return:
        a) The master device of the running process
        b) A list of GPU devices available to the running process
        c) Total number of GPUs available *globally*
        d) Total number of processes in the process group if there is one.
        """
        raise NotImplementedError

    @abstractmethod
    def parallelize_model(self, model):
        """
        Implements the logic of how to parallelize a model
        e.g. wrapping it in a DataParallel class
        """
        raise NotImplementedError

    def map_modules(self, modules, *args, **kwargs):
        """
        Given an arbitrary object of torch modules or tensors, it moves them to
        the master device of the calling process
        """
        return all_tensors_to(modules, device=self.device, *args, **kwargs)

    def get_master_device(self):
        """
        Returns the master device object
        """
        return self.device
