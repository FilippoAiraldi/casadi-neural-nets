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
