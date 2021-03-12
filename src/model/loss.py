"""
Module containing all loss functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyLogits(nn.Module):
    """
    Binary cross entropy for binary classification problem
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, output, target):  # pylint: disable=arguments-differ
        target = target['target'].float()
        return self.loss(output, target)


class FocalLoss(nn.Module):
    """
    Focal loss

    Adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """
    def __init__(self, alpha=(0.5, 0.5), gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        alpha = torch.tensor(list(alpha))  # pylint: disable=not-callable
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):  # pylint: disable=arguments-differ
        targets = targets["target"]
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction="none")
        p_t = torch.exp(-bce_loss)
        f_loss = self.alpha[targets] * (1 - p_t)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class SoftMarginFocalLoss(nn.Module):
    def __init__(self, weight_pos=2, weight_neg=1, gamma=2, margin=0.2):
        super().__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.gamma = gamma
        self.margin = margin

    def forward(self, logit, truth):
        # Soft-margin focal loss
        truth = truth["target"]
        margin = self.margin
        gamma = self.gamma
        weight_pos = self.weight_pos
        weight_neg = self.weight_neg
        em = np.exp(margin)

        logit = logit.view(-1)
        truth = truth.view(-1)
        log_pos = -F.logsigmoid(logit)
        log_neg = -F.logsigmoid(-logit)

        log_prob = truth * log_pos + (1 - truth) * log_neg
        prob = torch.exp(-log_prob)
        margin = torch.log(em + (1 - em) * prob)

        weight = truth * weight_pos + (1 - truth) * weight_neg
        loss = margin + weight * (1 - prob)**gamma * log_prob

        loss = loss.mean()
        return loss


class PairwiseRankingLoss(nn.Module):
    def forward(self, logit, truth):
        truth = truth["target"]
        num = len(truth)
        pos = logit[truth > 0.5]
        neg = logit[truth < 0.5]
        pos = pos.view(-1, 1)
        neg = neg.view(1, -1)
        diff = (pos - neg).view(-1)
        loss = -F.logsigmoid(diff)
        loss = loss.sum() / num
        return loss


class BCEPlusRank(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BinaryCrossEntropyLogits()
        self.pair = PairwiseRankingLoss()

    def forward(self, logit, truth):
        loss1 = self.bce(logit, truth)
        loss2 = self.pair(logit, truth)
        return (loss1 + loss2) / 2


def range_to_anchors_and_delta(precision_range, num_anchors):
    """Calculates anchor points from precision range.
        Args:
            precision_range: an interval (a, b), where 0.0 <= a <= b <= 1.0
            num_anchors: int, number of equally spaced anchor points.
        Returns:
            precision_values: A `Tensor` of [num_anchors] equally spaced values
                in the interval precision_range.
            delta: The spacing between the values in precision_values.
        Raises:
            ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if len(precision_range) != 2:
        raise ValueError("length of precision_range (%d) must be 2" % len(precision_range))
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError("precision values must follow 0 <= %f <= %f <= 1" % (precision_range[0], precision_range[1]))

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = np.linspace(start=precision_range[0], stop=precision_range[1], num=num_anchors + 1)[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return torch.tensor(precision_values), delta


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)


def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    """
    Args:
        labels: one-hot representation `Tensor` of shape broadcastable to logits
        logits: A `Tensor` of shape [N, C] or [N, C, K]
        positive_weights: Scalar or Tensor
        negative_weights: same shape as positive_weights
    Returns:
        3D Tensor of shape [N, C, K], where K is length of positive weights
        or 2D Tensor of shape [N, C]
    """
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    negative_weights_is_tensor = torch.is_tensor(negative_weights)

    # Validate positive_weights and negative_weights
    if positive_weights_is_tensor ^ negative_weights_is_tensor:
        raise ValueError("positive_weights and negative_weights must be same shape Tensor "
                         "or both be scalars. But positive_weight_is_tensor: %r, while "
                         "negative_weight_is_tensor: %r" % (positive_weights_is_tensor, negative_weights_is_tensor))

    if positive_weights_is_tensor and (positive_weights.size() != negative_weights.size()):
        raise ValueError("shape of positive_weights and negative_weights "
                         "must be the same! "
                         "shape of positive_weights is {0}, "
                         "but shape of negative_weights is {1}" % (positive_weights.size(), negative_weights.size()))

    # positive_term: Tensor [N, C] or [N, C, K]
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)

    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (positive_term.unsqueeze(-1) * positive_weights + negative_term.unsqueeze(-1) * negative_weights)
    else:
        return positive_term * positive_weights + negative_term * negative_weights


def build_class_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.
    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors


class AUCPRHingeLoss(nn.Module):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """
    def __init__(self, num_classes=1, num_anchors=20):
        """Args:
            config: Config containing `precision_range_lower`, `precision_range_upper`,
                `num_classes`, `num_anchors`
        """
        super().__init__()

        precision_range_lower = 0.0
        precision_range_upper = 1.0

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.precision_range = (
            precision_range_lower,
            precision_range_upper,
        )

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        precision_values, self.delta = range_to_anchors_and_delta(self.precision_range, self.num_anchors)
        self.register_buffer("precision_values", precision_values)

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(torch.zeros(self.num_classes, self.num_anchors, dtype=torch.float))
        self.lambdas = nn.Parameter(torch.ones(self.num_classes, self.num_anchors, dtype=torch.float))

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        targets = targets["target"]
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError("num classes is %d while logits width is %d" % (self.num_classes, C))

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(logits, targets, weights=weights)

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = lagrange_multiplier(self.lambdas)
        # print("lambdas: {}".format(lambdas))

        # A `Tensor` of Shape [N, C, K]
        hinge_loss = weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values,
        )

        # 1D tensor of shape [C]
        class_priors = build_class_priors(labels, weights=weights)

        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (lambdas * (1.0 - self.precision_values))

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        loss = per_anchor_loss.sum(2) * self.delta
        loss = loss / (self.precision_range[1] - self.precision_range[0])

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        # Converts targets to one-hot representation. Dim: [N, C]
        labels = torch.zeros((N, C), dtype=torch.float, device=logits.device).scatter(1, targets.unsqueeze(1), 1)

        if weights is None:
            weights = torch.ones(N, dtype=torch.float, device=logits.device)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        return labels, weights


class MacroSoftF1_v1(nn.Module):
    def forward(self, y_hat, y):
        """Compute the macro soft F1-score as a cost.
        Average (1 - soft-F1) across all labels.
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = y["target"]
        y = y.float()
        y_hat = y_hat.float()
        tp = torch.sum(y_hat * y, dim=0)
        fp = torch.sum(y_hat * (1 - y), dim=0)
        fn = torch.sum((1 - y_hat) * y, dim=0)
        soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = cost.mean()  # average on all labels

        return macro_cost


class MacroSoftF1_v2(nn.Module):
    def forward(self, y_hat, y):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = y["target"]
        y = y.float()
        y_hat = y_hat.float()
        tp = torch.sum(y_hat * y, dim=0)
        fp = torch.sum(y_hat * (1 - y), dim=0)
        fn = torch.sum((1 - y_hat) * y, dim=0)
        tn = torch.sum((1 - y_hat) * (1 - y), dim=0)
        soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
        cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
        macro_cost = cost.mean()  # average on all labels
        return macro_cost


class MSELoss2(nn.Module):
    """
    Wrapper around MSE loss
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):  # pylint: disable=arguments-differ
        target = target["target"].float()
        return self.mse(output, target)


class PoissonRegressionLoss(nn.Module):
    """
    Poisson regression loss derived as MLE of poisson distribution
    """
    def forward(self, output, target):  # pylint: disable=arguments-differ
        # Assume output is ln lambda (ln of the poisson rate)
        target = target["target"].float()
        loss = torch.exp(output) - target * output
        return loss.mean()
