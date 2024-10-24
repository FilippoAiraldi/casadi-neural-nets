import csnn.functional as F
from csnn.module import Module, SymType


class ReLU(Module[SymType]):
    """Applies the rectified linear unit function element-wise as
    `ReLU(x) = (x)^+ = max(0, x)`."""

    def forward(self, input: SymType) -> SymType:
        return F.relu(input)


class SoftPlus(Module[SymType]):
    """Applies the softplus function element-wise as
    `Softplus(x) = 1 / beta * log(1 + exp(beta * x))`."""

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
