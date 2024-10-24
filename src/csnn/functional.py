from itertools import product
from random import random
from typing import Optional, TypeVar

import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)


x = cs.MX.sym("x_mx")
y = cs.logsumexp(cs.vertcat(x, 0))
softplus_elementwise_mx = cs.Function("softplus_mx", [x], [y])
del x, y


def linear(input: SymType, weight: SymType, bias: Optional[SymType] = None) -> SymType:
    """Applies a linear transformation to the incoming data: `y = xA^T + b`."""
    output = input @ weight.T
    if bias is not None:
        output = (output.T + bias.T).T  # transpose trick is required
    return output


def relu(input: SymType) -> SymType:
    """Applies the rectified linear unit function element-wise as
    `ReLU(x) = (x)^+ = max(0, x)`."""
    return cs.fmax(0, input)


def softplus(input: SymType, beta: float = 1.0, threshold: float = 20.0) -> SymType:
    """Applies the softplus function element-wise as
    `Softplus(x) = 1/beta * log(1 + exp(beta * x))`."""
    bi = beta * input
    if isinstance(input, cs.SX):
        return cs.if_else(input > threshold, bi, cs.log1p(cs.exp(bi)) / beta)
    return softplus_elementwise_mx(bi) / beta


def sigmoid(input: SymType) -> SymType:
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`."""
    return 1 / (1 + cs.exp(-input))


def tanh(input: SymType) -> SymType:
    """Applies the element-wise function
    `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`."""
    return cs.tanh(input)


def dropout(input: SymType, p: float = 0.5, training: bool = False) -> SymType:
    """Randomly zeroes some of the elements of the input tensor with probability `p`.
    Conversely to PyTorch, this function is always inplace."""
    if not training:
        return input
    n, m = input.shape  # SX/MX inputs are always 2D
    for i, j in product(range(n), range(m)):
        if random() < p:
            input[i, j] = 0
    return input / (1 - p)


def dropout1d(input: SymType, p: float = 0.5, training: bool = False) -> SymType:
    """Randomly zeroes some of the channles of the input tensor with probability `p`.
    Conversely to PyTorch, this function is always inplace."""
    if not training:
        return input
    n = input.shape[0]  # SX/MX inputs are always 2D
    for i in range(n):
        if random() < p:
            input[i, :] = 0
    return input / (1 - p)
