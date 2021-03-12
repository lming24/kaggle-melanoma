"""Module that contains all device mappers"""

import torch
import torch.cuda
import torch.multiprocessing
from base.base_device_mapper import BaseDeviceMapper
from lib.data_parallel import CustomDataParallel


class SimpleDataParallel(BaseDeviceMapper):
    """
    Device mapper that initializes the number of requested GPUs if available.
    Applies data parallelism if more than one GPUs is available.
    """
    def prepare_device(self):
        """
        setup GPU device if available, move model into configured device and
        enable data parallelism if more than 1 GPUs are available.
        """
        # Number of GPU requested
        n_gpu_use = self.n_gpu

        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There's no GPU available on this machine,"
                                "training will be performed on the CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is %s, "
                "but only %s are available on this machine.", n_gpu_use, n_gpu)
            n_gpu_use = n_gpu

        # Use the first available GPU as master or CPU
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

        list_ids = list(range(n_gpu_use))
        return device, list_ids, len(list_ids), 1

    def parallelize_model(self, model):
        if not self.auto_parallel:
            return model
        model = model.to(self.device)
        if self.n_gpu > 1:
            model = CustomDataParallel(model, device_ids=self.gpu_ids)
        return model
