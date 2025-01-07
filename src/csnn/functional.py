from itertools import product
from random import random
from typing import Literal, Optional, TypeVar

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


def leaky_relu(input: SymType, negative_slope: float = 0.01) -> SymType:
    """Applies the leaky rectified linear unit function element-wise as
    `LeakyReLU(x) = max(0, x) + alpha * min(0, x)`, where `alpha` is a small
    positive slope, by default `0.01`."""
    return cs.fmax(0, input) + negative_slope * cs.fmin(0, input)


def gelu(input: SymType, approximate: Literal["none", "tanh"] = "none") -> SymType:
    """Applies the element-wise function
    `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, possibly approximated to
    `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 x^3)))`."""
    if approximate == "none":
        return 0.5 * input * (1 + cs.erf(input / cs.sqrt(2)))
    x3 = cs.power(input, 3)
    return 0.5 * input * (1 + tanh(cs.sqrt(2 / cs.pi) * (input + 0.044715 * x3)))


def elu(input: SymType, alpha: float = 1.0) -> SymType:
    """Applies the ELU function element-wise as
    `ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))`."""
    return cs.fmax(0, input) + cs.fmin(0, alpha * cs.expm1(input))


def selu(input: SymType) -> SymType:
    """Applies the SELU function element-wise as
    `SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`, where `alpha`
    and `scale` are pre-defined constants."""
    return 1.0507009873554804934193349852946 * elu(
        input, 1.6732632423543772848170429916717
    )


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


def rnn_cell(
    input: SymType,
    hidden: SymType,
    weight_ih: SymType,
    weight_hh: SymType,
    bias_ih: Optional[SymType] = None,
    bias_hh: Optional[SymType] = None,
    nonlinearity: Literal["tanh", "relu"] = "tanh",
) -> SymType:
    """Computes the output of a single Elman RNN cell."""
    out = linear(input, weight_ih, bias_ih) + linear(hidden, weight_hh, bias_hh)
    if nonlinearity == "tanh":
        return tanh(out)
    return relu(out)


def rnn(
    input: SymType,
    hidden: SymType,
    weights_ih: list[SymType],
    weights_hh: list[SymType],
    biases_ih: Optional[list[SymType]] = None,
    biases_hh: Optional[list[SymType]] = None,
    nonlinearity: Literal["tanh", "relu"] = "tanh",
) -> tuple[SymType, SymType]:
    """Applies a multi-layer Elman RNN cell with tanh or ReLU nonlinearity."""
    num_layers, h_size = hidden.shape
    seq_len, in_size = input.shape
    has_biases = biases_ih is not None

    # transform the evaluation of all layers into a single function call - but if the
    # sequence length is 1, we can just return at the end of the loop
    if seq_len == 1:
        input_ = input[0, :]
        hidden_ = hidden
    else:
        sym = weights_ih[0].sym
        input_ = input_loop = sym("in", in_size, 1).T
        hidden_ = sym("hidd", *hidden.shape)
    output_ = []
    for layer in range(num_layers):
        input_loop = rnn_cell(
            input_loop,
            hidden_[layer, :],
            weights_ih[layer],
            weights_hh[layer],
            biases_ih[layer] if has_biases else None,
            biases_hh[layer] if has_biases else None,
            nonlinearity,
        )
        output_.append(input_loop)
    output = cs.vcat(output_)
    if seq_len == 1:
        return output_[-1], output

    weights = weights_ih + weights_hh
    if has_biases:
        weights += biases_ih + biases_hh
    layers = cs.Function(
        "L", [hidden_, input_.T] + weights, (output, output[-1, :].T), {"cse": True}
    )

    # process each layer
    # OLD FOR-LOOP IMPLEMENTATION
    # output_.clear()
    # for t in range(seq_len):
    #     hidden, output = layers(hidden, input[t, :].T, *weights)
    #     output_.append(output.T)
    # return cs.vcat(output_), hidden
    mapaccum = layers.mapaccum(seq_len)
    all_hiddens, output = mapaccum(hidden, input.T, *weights)
    last_hidden = all_hiddens[:, -h_size:]
    return output.T, last_hidden


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
