from collections.abc import Iterator, Sequence
from typing import Optional, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..activation import Softplus
from ..containers import Sequential
from ..init import RngType, init_parameters
from ..linear import Linear
from ..module import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class FicnnLayer(Module[SymType]):
    """A fully connected input convex neural network (FICNN) layer consisting of two
    linear elements."""

    def __init__(
        self,
        in_features: int,
        prev_hidden_features: int,
        out_features: int,
        act: Optional[type[Module[SymType]]] = None,
    ) -> None:
        """Creates a layer of a FICNN.

        Parameters
        ----------
        in_features : int
            Feature size of the very first input of the convex net.
        hidden_features : int
            Feature size of the previous hidden layer's output.
        out_features : int
            Feature size of this layer's output.
        act : Callable, optional
            Class of the activation function, optional.
        """
        super().__init__()
        self.y_layer = Linear(in_features, out_features, bias=True)
        self.z_layer = Linear(prev_hidden_features, out_features, bias=False)
        self.act = act() if act is not None else None

    def forward(self, input: tuple[SymType, SymType]) -> tuple[SymType, SymType]:
        y, z = input
        z_new = self.y_layer(y) + self.z_layer(z)
        if self.act is not None:
            z_new = self.act(z_new)
        return y, z_new


class FicNN(Module[SymType]):
    """Feed-forward, fully connected, fully input convex neural network (FICNN) [1].

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : sequence of ints
        Number of features in each hidden layer.
    out_features : int
        Number of output features.
    act : type of activation function, optional
        Class of the activation function. By default, `Softplus` is used.

    Raises
    ------
    ValueError
        Raises if the number of hidden layers is less than 1.

    References
    ----------
    [1] Amos, B., Xu, L. and Kolter, J.Z., 2017, July. Input convex neural networks. In
        International Conference on Machine Learning (pp. 146-155). PMLR.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        act: type[Module[SymType]] = Softplus,
    ) -> None:
        super().__init__()
        if len(hidden_features) < 1:
            raise ValueError("FICNN must have at least one hidden layer")
        self.in_features = in_features
        self.input_linear = Linear(in_features, hidden_features[0], True)
        self.input_act = act()
        self.hidden_layers = Sequential(
            FicnnLayer(in_features, hidden_features[i], hidden_features[i + 1], act)
            for i in range(len(hidden_features) - 1)
        )
        self.last_layer = FicnnLayer(in_features, hidden_features[-1], out_features)

    def forward(self, input: SymType) -> SymType:
        z1 = self.input_act(self.input_linear(input))
        _, zf_1 = self.hidden_layers((input, z1))  # i.e., z_{f-1}
        _, zf = self.last_layer((input, zf_1))
        return zf

    def init_parameters(
        self,
        enforce_convex: bool = True,
        prefix: str = "",
        seed: RngType = None,
    ) -> Iterator[tuple[str, npt.NDArray[np.floating]]]:
        """Similarly to `csnn.init_parameters`, initializes the parameters
        (i.e., weights) of this neural network in such a way to enforce convexity. This
        is achieved by setting the weights of all `z_layers` to be nonnegative. If
        this property is to be preserved, then make sure to keep these weights
        nonnegative.

        Parameters
        ----------
        enforce_convex : bool, optional
            Whether to enforce the function to be convex, by default `True`.
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
            if enforce_convex and name.endswith("z_layer.weight"):
                par = np.abs(par)
            yield name, par
