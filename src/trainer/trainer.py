"""
Module that contains parts of the training loop. In particular, the logic
of how to train a single train epoch and a single valid epoch.
"""

from sklearn.metrics import roc_auc_score
import numpy as np
import torch
# from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from lib.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
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
                 monitor,
                 early_stop,
                 tensorboard,
                 len_epoch=None,
                 add_roc_auc=True,
                 softmax=False):
        # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
        super().__init__(config, device_mapper, model, loss, metrics, optimizer, data_loader, valid_data_loader,
                         lr_scheduler, save_period, epochs, monitor, early_stop, tensorboard)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.softmax = softmax

        self.train_metrics = MetricTracker('loss', *[m.name() for m in self.metrics], writer=self.writer)

        self.add_roc_auc = add_roc_auc
        if add_roc_auc:
            self.valid_metrics = MetricTracker('loss', 'roc_auc', *[m.name() for m in self.metrics], writer=self.writer)
        else:
            self.valid_metrics = MetricTracker('loss', *[m.name() for m in self.metrics], writer=self.writer)

    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        device_mapper = self.device_mapper
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            data = device_mapper.map_modules(data, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, data)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), batch_size=output.size(0))
            for met in self.metrics:
                self.train_metrics.update(met.name(), met(output, data), batch_size=output.size(0))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: %d %s Loss: %.6f', epoch, self._progress(batch_idx), loss.item())

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self.valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(log["val_roc_auc"])
            else:
                self.lr_scheduler.step()
        return log

    def valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        device_mapper = self.device_mapper
        self.model.eval()
        self.valid_metrics.reset()
        targets = []
        outputs = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                data = device_mapper.map_modules(data, non_blocking=True)

                output = self.model(data)
                loss = self.loss(output, data)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), batch_size=output.size(0))
                for met in self.metrics:
                    self.valid_metrics.update(met.name(), met(output, data), batch_size=output.size(0))

                if self.add_roc_auc:
                    if not self.softmax:
                        outputs.append(torch.sigmoid(output).cpu().numpy().reshape(-1))
                    else:
                        outputs.append(torch.nn.functional.softmax(output, dim=1)[:, 1].cpu().numpy().reshape(-1))
                    targets.append(data['target'].cpu().numpy().reshape(-1))

        if self.add_roc_auc:
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)
            score = roc_auc_score(targets, outputs)
            # size doesn't matter here, it's a global metric not per batch
            self.valid_metrics.update('roc_auc', score, batch_size=1)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
