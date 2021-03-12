"""
Modules containing all metric functions used during training or validation
"""

import torch
import torch.nn as nn
from base.base_metric import BaseMetric


class AccuracyWithLogits(BaseMetric):
    """
    Calculates the accuracy of `output` given the correct predictions `target`

    :param output: Tensor of size (N, 1) where N is the number of samples. It contains
        probability of class 1. >=0.5 counts as class 1.
    :param target: Tensor of size (N, 1). Contains the correct class number
    :return: Accuracy level
    """
    def forward(self, output, target):
        with torch.no_grad():  # Do not calculate gradient for metrics
            target = target['target']
            if output.dim() == 1 or output.size(1) == 1:
                pred = torch.round(torch.sigmoid(output)).byte()
            else:
                pred = torch.nn.functional.softmax(output, dim=1)[:, 1]
                pred = torch.round(pred).byte()
            assert pred.size(0) == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)

    def name(self):
        return 'accuracy'


class MSEMetricFromCounts(BaseMetric):
    """
    Wrapper around MSE loss
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):  # pylint: disable=arguments-differ
        with torch.no_grad():
            target = target["target"].float()
            loss = self.mse(output, target)
        return loss.item()

    def name(self):
        return "mse"


class MSEMetricFromLogLambda(BaseMetric):
    """
    Wrapper around MSE loss. Network output is assumed to be log lambda where
    lambda is the parameter of the poisson distribution
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):  # pylint: disable=arguments-differ
        with torch.no_grad():
            target = target["target"].float()
            loss = self.mse(torch.exp(output), target)
        return loss.item()

    def name(self):
        return "mse"


class PoissonModeAccuracy(BaseMetric):
    """
    Frequency that the mode of a poisson distribution is equal to the correct
    value. Input is log lambda where lambda is the poisson rate
    """
    def forward(self, output, target):  # pylint: disable=arguments-differ
        with torch.no_grad():
            target = target["target"]
            output = torch.floor(torch.exp(output))  # mode of poisson assuming output is rate lambda
            loss = (output == target).float().mean()
        return loss.item()

    def name(self):
        return "mode_accuracy"


class RoundedAccuracy(BaseMetric):
    """
    Round the output and calculate the percentage we are correct
    """
    def forward(self, output, target):  # pylint: disable=arguments-differ
        with torch.no_grad():
            target = target["target"]
            output = torch.round(output)  # mode of poisson assuming output is rate lambda
            loss = (output == target).float().mean()
        return loss.item()

    def name(self):
        return "rounded_accuracy"


class ConstantPredictionMSE(BaseMetric):
    """
    MSE of a model that always predicts 50% of the bag is malignant
    """
    def forward(self, output, target):
        target = target["target"].float() - target["len"].float() / 2
        target = target * target
        return target.mean().item()

    def name(self):
        return 'const_pred'
