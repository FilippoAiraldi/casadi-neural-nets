from collections.abc import Iterator
from typing import TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..activation import ReLU
from ..init import RngType, init_parameters
from ..linear import Linear
from ..module import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class ElementWiseSquare(Module):
    """Squares the input in an element-wise fashion."""

    def forward(self, x: SymType) -> SymType:
        return x * x


class DotProduct(Linear):
    """Compute the dot product of the input with some weights."""

    def __init__(self, in_features: int) -> None:
        super().__init__(in_features, 1, False)


class PwqNN(Module[SymType]):
    """Piecewise quadratic (PWQ) neural network [1].

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of features in the single hidden layer.
    act : type of activation function, optional
        Class of the activation function. By default, `ReLU` is used.

    References
    ----------
    [1] He, K., Shi, S., Boom, T.V.D. and De Schutter, B., 2022. Approximate Dynamic
        Programming for Constrained Linear Systems: A Piecewise Quadratic Approximation
        Approach. arXiv preprint arXiv:2205.10065.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act: type[Module[SymType]] = ReLU,
    ) -> None:
        super().__init__()
        self.input_layer = Linear(in_features, hidden_features)
        self.activation = act()
        self.element_wise_sq = ElementWiseSquare()
        self.dot = DotProduct(hidden_features)

    def forward(self, x: SymType) -> SymType:
        return self.dot(self.element_wise_sq(self.activation(self.input_layer(x))))

    def init_parameters(
        self,
        enforce_convex: bool = True,
        enforce_zero_at_origin: bool = True,
        prefix: str = "",
        seed: RngType = None,
    ) -> Iterator[tuple[str, npt.NDArray[np.floating]]]:
        """Similarly to `csnn.init_parameters`, initializes the parameters
        (i.e., weights) of this neural network in such a way to enforce convexity and
        zero output near the origin. This is achieved by setting the weights of the
        dot product layer to be nonnegative (i.e., the output is a positive combination
        of squares, so it is convex), and the bias of the input layer to be nonpositve
        (it can be shown this induces small values near the origin). If these properties
        are to be preserved, then make sure to keep these weights nonnegative and
        nonpositive, respectively.

        Parameters
        ----------
        enforce_convex : bool, optional
            Whether to enforce the function to be convex, by default `True`.
        enforce_zero_at_origin : bool, optional
            Whether to enforce the value of the function at the origin to be close to
            zero, by default `True`.
        prefix : str, optionals
            Prefix to prepend to the names of the parameters, by default "".
        seed : int, sequence of ints, or rng engine, optional
            Seed for the random number generator, or an engine itself, by default
            `None`.

        Yields
        ------
        Iterator of 2-tuples of str and arrays
            Yields the name and value of each parameter of the neural network.
        """
        pars = init_parameters(self, True, prefix, True, seed)
        for name, par in pars:
            # force convexity of nn function
            if enforce_convex and name.endswith("dot.weight"):
                par = np.abs(par)
            # force value at origin to be close to zero
            elif enforce_zero_at_origin and name.endswith("input_layer.bias"):
                par = -np.abs(par)
            yield name, par
