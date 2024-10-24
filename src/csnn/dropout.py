import csnn.functional as F
from csnn.module import Module, SymType


class Dropout(Module[SymType]):
    """Randomly zeroes some of the elements of the input tensor with probability `p`.
    Use `random.seed` to ensure reproducibility.

    Parameters
    ----------
    p : float, optional
        Probability of an element to be zeroed, by default `0.5`.
    training : bool, optional
        Whether the module is in training mode or not, by default `False`. If not in
        training mode, the module simply returns the input.

    Notes
    -----
    This layer is implemented for compatibility with PyTorch, but it makes no sense to
    use when building a symbolic representation of the neural network, as the dropout
    indices are randomly generated at instantiation only once, and then kept fixed.
    """

    def __init__(self, p: float = 0.5, training: bool = False) -> None:
        super().__init__()
        self.p = p
        self.training = training

    def forward(self, input: SymType) -> SymType:
        return F.dropout(input, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}, training={self.training}"


class Dropout1d(Module[SymType]):
    """Randomly zeroes some of the channels of the input tensor with probability `p`.
    Use `random.seed` to ensure reproducibility.

    Parameters
    ----------
    p : float, optional
        Probability of an element to be zeroed, by default `0.5`.
    training : bool, optional
        Whether the module is in training mode or not, by default `False`. If not in
        training mode, the module simply returns the input.

    Notes
    -----
    This layer is implemented for compatibility with PyTorch, but it makes no sense to
    use when building a symbolic representation of the neural network, as the dropout
    indices are randomly generated at instantiation only once, and then kept fixed.
    """

    def __init__(self, p: float = 0.5, training: bool = False) -> None:
        super().__init__()
        self.p = p
        self.training = training

    def forward(self, input: SymType) -> SymType:
        return F.dropout1d(input, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}, training={self.training}"
