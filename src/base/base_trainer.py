"""
Module containing the training loop base class
"""

import os
import pathlib
from abc import ABC, abstractmethod

import torch
from numpy import inf

import lib.env
import logger.logger as logger
from logger.visualization import TensorboardWriter


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments

    def __init__(self,
                 config,
                 device_mapper,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 data_loader,
                 valid_data_loader,
                 lr_scheduler,
                 save_period,
                 epochs,
                 monitor=None,
                 early_stop=None,
                 tensorboard=True):
        self.logger = logger.get_logger('trainer', verbosity=2)

        self.config = config
        self.device_mapper = device_mapper
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.save_period = int(save_period)
        self.epochs = int(epochs)
        self.monitor = monitor

        # configuration to monitor model performance and save best
        if self.monitor is None:
            self.monitor_mode = None
            self.monitor_best = 0
            if early_stop is not None:
                self.logger.warning("'early_stop' attribute will be ignored because 'monitor' attribute is not set")
            early_stop = inf
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf
            if early_stop is None:
                early_stop = inf

        self.early_stop = early_stop

        self.start_epoch = 1

        self.checkpoint_dir = pathlib.Path(os.getenv(lib.env.CHECKPOINT_ENV))
        self.log_dir = pathlib.Path(os.getenv(lib.env.LOG_ENV))

        # setup visualization writer instance
        self.writer = TensorboardWriter(self.log_dir, tensorboard)

        if config.resume is not None:
            self.resume_checkpoint(config.resume)

    @abstractmethod
    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self.train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    %20s: %.4f', str(key), value)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode is not None:
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.monitor_mode == 'min' and log[self.monitor_metric] <= self.monitor_best) or \
                               (self.monitor_mode ==
                                'max' and log[self.monitor_metric] >= self.monitor_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '%s' is not found. "
                        "Model performance monitoring is disabled.", self.monitor_metric)
                    self.monitor_mode = None
                    improved = False

                if improved:
                    self.monitor_best = log[self.monitor_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for %s epochs. "
                                     "Training stops.", self.early_stop)
                    break

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, save_best=best)

    def save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: %s ...", filename)
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: %s ...", resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch %s", self.start_epoch)
