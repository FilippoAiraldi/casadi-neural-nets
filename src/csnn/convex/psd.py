import sys
from collections.abc import Sequence
from itertools import chain
from typing import Literal, Optional, TypeVar

import casadi as cs
from numpy import broadcast_to, tril_indices, triu_indices
from numpy.typing import ArrayLike

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


def _reshape_mat(x: SymType, size: int, shape: Literal["triu", "tril"]) -> SymType:
    # casadi at most supports 2D matrices, so the input must be non-batched
    x = cs.vec(x)
    if shape == "triu":
        indices = triu_indices(size)
        ensure_triangular = cs.triu
    else:
        indices = tril_indices(size)
        ensure_triangular = cs.tril

    mat = x.zeros(size, size)
    k = 0
    for i, j in zip(*indices):
        mat[i, j] = x[k]
        k += 1
    return ensure_triangular(mat)


class PsdNN(Module[SymType]):
    """Network that computes a quadratic form by returning the Cholesky decomposition
    of a positive semi-definite matrix (PSD) and a reference point for any input. The
    decomposition is returned as a flattened triangular matrix, by default; however,
    when calling the model one can choose also to return the upper or lower triangular
    part of the matrix (see `forward`) for more details.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : sequence of ints
        Number of features in each hidden linear layer.
    out_size : int
        Size of the quadratic form elements (i.e., the side length of the PSD matrix).
    out_shape : {"flat", "triu", "tril"}
        Shape of the output PSD matrix. If "flat", the output is not reshaped in any
        matrix. If "triu" or "tril", the output is reshaped as an upper or lower
        triangular, but does not support batched inputs.
    act : type of activation function, optional
        Class of the activation function. By default, `ReLU` is used.
    eps : array-like, optional
        Value to add to the PSD matrix, e.g., to ensure it is positive definite. Should
        be broadcastable to the shape `(out_size, out_size)`. By default, an identity
        matrix with `1e-4` is used. Only used in the `quadform` method.

    Raises
    ------
    ValueError
        Raises if the number of hidden layers is less than 1.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_size: int,
        out_shape: Literal["flat", "triu", "tril"],
        act: type[Module[SymType]] = ReLU,
        eps: Optional[ArrayLike] = None,
    ) -> None:
        if len(hidden_features) < 1:
            raise ValueError("Psdnn must have at least one hidden layer")
        super().__init__()
        features = chain([in_features], hidden_features)
        self.hidden_layers = Sequential(
            chain.from_iterable((Linear(i, j), act()) for i, j in pairwise(features))
        )
        self.mat_head = Linear(hidden_features[-1], (out_size * (out_size + 1)) // 2)
        self.ref_head = Linear(hidden_features[-1], out_size)
        self._eps = (
            1e-4 * cs.DM_eye(out_size)
            if eps is None
            else cs.DM(broadcast_to(eps, (out_size, out_size)))
        )
        self._out_shape = out_shape

    def _forward(self, input: SymType) -> tuple[SymType, SymType]:
        """Forward pass without reshaping the output matrix"""
        h = self.hidden_layers(input)
        ref = self.ref_head(h)
        mat_flat = self.mat_head(h)
        return mat_flat, ref

    def forward(self, input: SymType) -> tuple[SymType, SymType]:
        """Forward pass folloewd by reshaping of the output matrix"""
        mat, ref = self._forward(input)
        if self._out_shape != "flat":
            mat = _reshape_mat(mat, self._eps.size1(), self._out_shape)
        return mat, ref

    def quadform(self, x: SymType, context: SymType) -> SymType:
        """Computes the quadratic form `(x - x_ref)' Q (x - x_ref)` where `Q` is the
        predicted the PSD matrix and `x_ref` the reference point.

        Parameters
        ----------
        x : SymType
            The value at which the quadratic form is evaluated.
        context : SymType
            The context passed as input to the neural network for the prediction of the
            PSD matrix and the reference point.

        Returns
        -------
        SymType
            The value of the quadratic form.
        """
        L_flat, ref = self._forward(context)
        L = _reshape_mat(L_flat, self._eps.size1(), "tril")
        return cs.bilin(L @ L.T + self._eps, x - ref)
