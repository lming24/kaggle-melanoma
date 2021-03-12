"""LR Schedulers"""

from torch.optim.lr_scheduler import _LRScheduler


class LRRampUp(_LRScheduler):
    """
    Start small, increase and then exponential decay
    """
    def __init__(self,
                 optimizer,
                 lr_max=0.00005,
                 lr_min=0.00001,
                 lr_rampup_epochs=5,
                 lr_sustain_epochs=0,
                 lr_exp_decay=0.8,
                 last_epoch=-1):
        # pylint: disable=too-many-arguments
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_rampup_epochs = lr_rampup_epochs
        self.lr_sustain_epochs = lr_sustain_epochs
        self.lr_exp_decay = lr_exp_decay
        super(LRRampUp, self).__init__(optimizer, last_epoch)

    def _lrfn(self, epoch, base_lr):
        # pylint: disable=invalid-name
        if epoch < self.lr_rampup_epochs:
            lr = (self.lr_max - base_lr) / self.lr_rampup_epochs * epoch + base_lr
        elif epoch < self.lr_rampup_epochs + self.lr_sustain_epochs:
            lr = self.lr_max
        else:
            lr = (self.lr_max - self.lr_min) * self.lr_exp_decay**(epoch - self.lr_rampup_epochs -
                                                                   self.lr_sustain_epochs) + self.lr_min
        return lr

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [self._lrfn(self.last_epoch, base_lr) for base_lr in self.base_lrs]
