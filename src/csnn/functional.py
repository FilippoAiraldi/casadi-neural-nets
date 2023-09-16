from typing import Optional, TypeVar

import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)


x_sx = cs.SX.sym("x_sx")
x_mx = cs.MX.sym("x_mx")
softplus_elementwise_sx = cs.Function(
    "softplus_sx", [x_sx], [cs.logsumexp(cs.vertcat(x_sx, 0))]
)
softplus_elementwise_mx = cs.Function(
    "softplus_mx", [x_mx], [cs.logsumexp(cs.vertcat(x_mx, 0))]
)
del x_sx, x_mx


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


def softplus(input: SymType, beta: float = 1.0) -> SymType:
    """Applies the softplus function element-wise as
    `Softplus(x) = 1/beta * log(1 + exp(beta * x))`."""
    f = softplus_elementwise_sx if isinstance(input, cs.SX) else softplus_elementwise_mx
    return f(beta * input) / beta
