import sys
from collections.abc import Iterable, Sequence
from itertools import chain, repeat
from typing import Optional, TypeVar, Union

import casadi as cs

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


def _create_linear_layer(
    in_out: tuple[int, int], act: Optional[type[Module[SymType]]], bias: bool
) -> Iterable[Module[SymType]]:
    """Creates a linear layer optionally followed by an activation function."""
    yield Linear(*in_out, bias)
    if act is not None:
        yield act()


class Mlp(Module[SymType]):
    """Multilayer perceptron (MLP).

    Parameters
    ----------
    hidden_features : sequence of ints
        Number of features in each layer (input, hidden1, hidden2, ..., output).
    acts : type of activation function or a sequence of, optional
        The activation functions to be used for each layer. If a single activation
        is passed, it is used for all layers. If a sequence is passed, each activation
        is used for the corresponding layer. If `None`, no activation is
        used for that layer. By default, `ReLU` is used.
    biases : bool or sequence of, optional
        If set to `False`, the corresponding linear layers will not learn an additive
        bias. Defaults to `True`.

    Raises
    ------
    ValueError
        Raises if the number of layers is less than 1; or if `acts` or `biases` are
        lists with a different length from `features`.
    """

    def __init__(
        self,
        features: Sequence[int],
        acts: Union[
            Optional[type[Module[SymType]]], Sequence[Optional[type[Module[SymType]]]]
        ] = ReLU,
        biases: Union[bool, Sequence[bool]] = True,
    ) -> None:
        if len(features) < 1:
            raise ValueError("Mlp must have at least one layer.")

        n_hidden = len(features) - 1
        if acts is None or isinstance(acts, type):
            acts = repeat(acts, n_hidden)
        elif len(acts) != n_hidden:
            raise ValueError("Invalid number of activation functions provided.")

        if isinstance(biases, bool):
            biases = repeat(biases, n_hidden)
        elif len(biases) != n_hidden:
            raise ValueError("Invalid number of bias flags provided.")

        super().__init__()
        self.layers = Sequential(
            chain.from_iterable(
                map(_create_linear_layer, pairwise(features), acts, biases)
            )
        )

    def forward(self, input: SymType) -> SymType:
        return self.layers(input)
