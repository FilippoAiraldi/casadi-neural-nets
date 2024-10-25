__version__ = "1.0.3"

__all__ = [
    "Dropout",
    "Dropout1d",
    "GELU",
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
    "get_sym_type",
    "set_sym_type",
]


from typing import Literal, Union

import casadi as cs

from csnn.activation import GELU, SELU, LeakyReLU, ReLU, Sigmoid, Softplus, Tanh
from csnn.containers import Sequential
from csnn.dropout import Dropout, Dropout1d
from csnn.linear import Linear
from csnn.module import Module
from csnn.recurrent import RNN, RNNCell


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
