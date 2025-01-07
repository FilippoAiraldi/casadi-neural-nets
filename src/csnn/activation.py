from typing import Literal

import csnn.functional as F
from csnn.module import Module, SymType


class ReLU(Module[SymType]):
    """Applies the rectified linear unit function element-wise as
    `ReLU(x) = (x)^+ = max(0, x)`."""

    def forward(self, input: SymType) -> SymType:
        return F.relu(input)


class LeakyReLU(Module[SymType]):
    """Applies the leaky rectified linear unit function element-wise as
    `LeakyReLU(x) = max(0, x) + alpha * min(0, x)`, where `alpha` is a small
    positive slope, by default `0.01`."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: SymType) -> SymType:
        return F.leaky_relu(input, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class GELU(Module[SymType]):
    """Applies the element-wise function
    `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, possibly approximated to
    `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 x^3)))`."""

    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: SymType) -> SymType:
        return F.gelu(input)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate}"


class ELU(Module[SymType]):
    """Applies the ELU function element-wise as
    `ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))`."""

    def forward(self, input: SymType) -> SymType:
        return F.elu(input)


class SELU(Module[SymType]):
    """Applies the SELU function element-wise as
    `SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`, where `alpha`
    and `scale` are pre-defined constants."""

    def forward(self, input: SymType) -> SymType:
        return F.selu(input)


class Softplus(Module[SymType]):
    """Applies the softplus function element-wise as
    `Softplus(x) = 1 / beta * log(1 + exp(beta * x))`.

    Parameters
    ----------
    beta : float, optional
        The beta parameter of the softplus function, by default `1.0`.
    threshold : float, optional
        The threshold parameter of the softplus function, by default `20.0`.
    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: SymType) -> SymType:
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


class Sigmoid(Module[SymType]):
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`."""

    def forward(self, input: SymType) -> SymType:
        return F.sigmoid(input)


class Tanh(Module[SymType]):
    """Applies the element-wise function
    `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`."""

    def forward(self, input: SymType) -> SymType:
        return F.tanh(input)
