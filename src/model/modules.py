"""
Modules used by models in model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishImplementation(torch.autograd.Function):
    """
    Implementation of Swish
    """
    @staticmethod
    def forward(ctx, i):  # pylint: disable=arguments-differ
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """
    Implementation of Swish
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        return SwishImplementation.apply(x)


class MILAttentionPool(nn.Module):
    """
    Multiple Instance learning pooling layer. This takes a bag of instances
    and uses a form of self attention to weight them and produce one representation
    for the whole bag
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, input_dim, hidden_dim, n_attentions, gated=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_attentions = n_attentions

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_dim, self.n_attentions)
        self.fc3 = None
        self.sigmoid = None
        if gated:
            self.fc3 = nn.Linear(self.input_dim, self.hidden_dim)
            self.sigmoid = nn.Sigmoid()

    def _per_instance(self, x):
        # pylint: disable=invalid-name
        A = self.fc1(x)
        A = self.tanh(A)
        if self.fc3:
            A = A * self.sigmoid(self.fc3(x))

        A = self.fc2(A)  # [batch*bag, n_attentions]
        return A

    def forward(self, x, bag_lengths):  # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        input_ = x  # [batch x bag, input_dim]

        first_bag_length = bag_lengths[0]
        all_same = torch.all(bag_lengths == first_bag_length).item()

        A = self._per_instance(x)

        if all_same:
            # Faster approach if all bags have same length
            bag_size = first_bag_length
            batch_size = bag_lengths.size(0)

            A = A.view(batch_size, bag_size, *A.size()[1:])
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=2)  # [batch, n_attentions, bag_size]

            x = torch.bmm(A, input_.view(batch_size, bag_size, -1))  # [batch, n_attentions, input_dim]
            x = x.view(batch_size, -1)
        else:
            bag_lengths = bag_lengths.tolist()

            A = torch.split(A, bag_lengths)  # list of bag x n_attentions

            x = torch.split(input_, bag_lengths)  # list of bag x input_dim

            x = [F.softmax(A_i.t(), dim=1) @ x_i for A_i, x_i in zip(A, x)]
            x = torch.stack(x)  # batch, n_attentions, input_dim
            x = torch.flatten(x, start_dim=1)

        return x
