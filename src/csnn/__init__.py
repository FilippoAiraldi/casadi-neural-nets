__version__ = "1.0.3rc1"

__all__ = [
    "Module",
    "Sequential",
    "Linear",
    "ReLU",
    "Sigmoid",
    "SoftPlus",
    "Tanh",
    "get_sym_type",
    "set_sym_type",
]


from typing import Literal, Union

import casadi as cs

from csnn.activation import ReLU, Sigmoid, SoftPlus, Tanh
from csnn.containers import Sequential
from csnn.linear import Linear
from csnn.module import Module


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
