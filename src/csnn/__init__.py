__version__ = "1.0.5.post2"

__all__ = [
    "Dropout",
    "Dropout1d",
    "GELU",
    "ELU",
    "LeakyReLU",
    "Module",
    "Sequential",
    "Linear",
    "ReLU",
    "RNN",
    "RNNCell",
    "SELU",
    "Sigmoid",
    "Softplus",
    "Tanh",
    "convex",
    "feedforward",
    "get_sym_type",
    "init_parameters",
    "set_sym_type",
]


from typing import Literal, Union

import casadi as cs

from .activation import ELU, GELU, SELU, LeakyReLU, ReLU, Sigmoid, Softplus, Tanh
from .containers import Sequential
from .dropout import Dropout, Dropout1d
from .linear import Linear
from .module import Module
from .recurrent import RNN, RNNCell


def get_sym_type() -> Union[type[cs.SX], type[cs.MX]]:
    """Gets the casadi's symbolic type used to build the networks.

    Returns
    -------
    type[cs.SX] or type[cs.MX]]
        The current symbolic type, either `casadi.SX` or `MX`.
    """
    return Module.sym_type


def set_sym_type(type: Literal["SX", "MX"]) -> None:
    """Sets the casadi's symbolic type to be used in building the networks.

    Parameters
    ----------
    type : "SX" or "MX"
        The name of the symbolic type to set.
    """
    Module.sym_type = getattr(cs, type)


# import these guys for last

import csnn.convex as convex
import csnn.feedforward as feedforward

from .init import init_parameters
