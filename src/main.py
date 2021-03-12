#!/usr/bin/env python3
"""
The main entry point to the training procedure. All necessary objects are
initialized here and the training procedure is started.
"""

import argparse
import collections

import numpy as np
import torch
import torch.nn.modules.loss as torch_loss

import data_loader.data_loaders as module_data
import logger.logger as module_logger
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer.scheduler as module_scheduler
import trainer.trainer as module_trainer
import dev_mapper.device_mapper as module_mapper
from lib.config_parser import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
# The following is also required for exact reproducibility. However they can
# have an impact on performance and are hence disabled by default.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def main():
    """
    Main function
    """
    logger = module_logger.get_logger('main')

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    # setup device mapper
    device_mapper = config.init_obj('device_mapper', module_mapper)

    # setup data_loader instances
    data_loader = config.init_obj('train_data_loader', module_data)
    valid_data_loader = config.init_obj('val_data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model = device_mapper.parallelize_model(model)
    logger.info(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    # get loss and metric modules
    loss = config.init_obj('loss', [module_loss, torch_loss])
    metrics = config.init_list_of_objs('metrics', module_metric)

    # Move loss and metrics parameters to correct GPU
    loss = device_mapper.map_modules(loss)
    metrics = device_mapper.map_modules(metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', [torch.optim.lr_scheduler, module_scheduler], optimizer)

    trainer = config.init_obj('trainer', module_trainer, config, device_mapper, model, loss, metrics, optimizer,
                              data_loader, valid_data_loader, lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    main()
