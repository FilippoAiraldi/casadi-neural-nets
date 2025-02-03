import sys
from collections.abc import Sequence
from itertools import chain
from typing import Literal, TypeVar

import casadi as cs
from numpy import tril_indices, triu_indices

if sys.version_info >= (3, 10):
    from itertools import pairwise
else:

    def pairwise(iterable):
        iterator = iter(iterable)
        a = next(iterator)
        for b in iterator:
            yield a, b
            a = b


from ..activation import ReLU
from ..containers import Sequential
from ..linear import Linear
from ..module import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class TriReshape(Module):
    """Layer that can reshape the input as a lower or upper triangular matrix."""

    def __init__(self, mat_size: int, mat_shape: Literal["triu", "tril"]) -> None:
        super().__init__()
        self.mat_size = mat_size
        self.mat_shape = mat_shape

    def forward(self, x: SymType) -> SymType:
        # casadi at most supports 2D matrices, so the input must be non-batched
        x = cs.vec(x)
        m = self.mat_size

        if self.mat_shape == "triu":
            indices = triu_indices(m)
            ensure_triangular = cs.triu
        else:
            indices = tril_indices(m)
            ensure_triangular = cs.tril

        mat = x.zeros(m, m)
        k = 0
        for i, j in zip(*indices):
            mat[i, j] = x[k]
            k += 1
        return ensure_triangular(mat)

    def extra_repr(self) -> str:
        return f"mat_size={self.mat_size}, mat_shape={self.mat_shape}"


class PsdNN(Module[SymType]):
    """Network that outputs the Cholesky decomposition of a positive semi-definite
    matrix (PSD) for any input. The decomposition is returned as a flattened
    triangular matrix, by default; however, when calling the model one can choose
    also to return the upper or lower triangular part of the matrix (see `forward`) for
    more details.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : sequence of ints
        Number of features in each hidden linear layer.
    out_mat_size : int
        Size of the output PSD square matrix (i.e., its side length).
    out_shape : {"flat", "triu", "tril"}
        Shape of the output matrix. If "flat", the output is not reshaped in any matrix.
        If "triu" or "tril", the output is reshaped as an upper or lower triangular,
        but does not support batched inputs.
    act : type of activation function, optional
        Class of the activation function. By default, `ReLU` is used.

    Raises
    ------
    ValueError
        Raises if the number of hidden layers is less than 1.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_mat_size: int,
        out_shape: Literal["flat", "triu", "tril"],
        act: type[Module[SymType]] = ReLU,
    ) -> None:
        if len(hidden_features) < 1:
            raise ValueError("Psdnn must have at least one hidden layer")
        super().__init__()
        features = chain([in_features], hidden_features)
        out_features = (out_mat_size * (out_mat_size + 1)) // 2
        inner_layers = chain.from_iterable(
            (Linear(i, j), act()) for i, j in pairwise(features)
        )
        last_layer = [Linear(hidden_features[-1], out_features)]
        if out_shape != "flat":
            last_layer.append(TriReshape(out_mat_size, out_shape))
        self.layers = Sequential(chain(inner_layers, last_layer))

    def forward(self, input: SymType) -> SymType:
        return self.layers(input)
