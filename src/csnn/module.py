from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Dict, Generic, Iterator, Optional, Tuple, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Module(ABC, Generic[SymType]):
    """Base class for all neural network modules. Your models should also subclass this
    class."""

    def __init__(self) -> None:
        """Initializes the module."""
        self.training: bool = False
        self._sym_parameters: Dict[str, SymType] = {}
        self._num_parameters: Dict[str, npt.NDArray[np.double]] = {}
        self._modules: Dict[str, "Module"] = {}

    def register_parameter(
        self, name: str, sym: SymType, val: Optional[npt.NDArray[np.double]] = None
    ) -> None:
        """Adds a parameter to the module.

        Parameters
        ----------
        name : str
            Name of the parameter.
        sym : SymType
            Symbol of the parameter.
        val : numpy array, optional
            Numerical value of the parameter. If `None`, an empty array is instanciated.

        Raises
        ------
        KeyError
            Raises if `name` is already in use.
        """
        if name in self._sym_parameters:
            raise KeyError(f"Parameter {name} already exists.")
        if val is None:
            val = np.empty(sym.shape, dtype=float)
        assert val.shape == sym.shape, "Incompatible shapes."
        self._sym_parameters[name] = sym
        self._num_parameters[name] = val

    def add_module(self, name: str, module: "Module") -> None:
        """Adds a child module to the current module.

        Parameters
        ----------
        name : str
            Name of the child module
        module : Module
            Child module to be added to this module.

        Raises
        ------
        KeyError
            Raises if `name` is already in use.
        """
        if name in self._modules:
            raise KeyError(f"Child module {name} already exists.")
        self._modules[name] = module

    def sym_parameters(
        self, recurse: bool = True, prefix: str = ""
    ) -> Iterator[Tuple[str, SymType]]:
        """Returns an iterator over the module's parameters.

        Parameters
        ----------
        recurse : bool, optional
            If `True`, then yields parameters of this module and all submodules.
            Otherwise, yields only parameters that are direct members of this module. By
            default `True`.
        prefix : str, optional
            Prefix to add in front of this module's name.

        Yields
        ------
        Iterator of tuple[str, casadi.SX or MX]
            An iterator over tuples of parameter's names and symbols.
        """
        if prefix != "":
            prefix += "."
        for name, par in self._sym_parameters.items():
            yield (prefix + name, par)
        if recurse:
            for name, module in self._modules.items():
                yield from module.sym_parameters(recurse=True, prefix=f"{prefix}{name}")

    def num_parameters(
        self, recurse: bool = True, prefix: str = ""
    ) -> Iterator[Tuple[str, Optional[npt.NDArray[np.double]]]]:
        """Returns an iterator over the module's parameters.

        Parameters
        ----------
        recurse : bool, optional
            If `True`, then yields parameters of this module and all submodules.
            Otherwise, yields only parameters that are direct members of this module. By
            default `True`.
        prefix : str, optional
            Prefix to add in front of this module's name.

        Yields
        ------
        Iterator of tuple[str, casadi.SX or MX]
            An iterator over tuples of parameter's names and numerical values.
        """
        if prefix != "":
            prefix += "."
        for name, par in self._num_parameters.items():
            yield (prefix + name, par)
        if recurse:
            for name, module in self._modules.items():
                yield from module.num_parameters(recurse=True, prefix=f"{prefix}{name}")

    @abstractmethod
    def forward_sym(self, x: SymType) -> SymType:
        """Forwards symbolically the given input through the neural net.

        Parameters
        ----------
        x : SymType
            Symbolical input.

        Returns
        -------
        SymType
            The symbolical output of the net.
        """

    @abstractmethod
    def forward_num(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        """Forwards numerically the given input through the neural net.

        Parameters
        ----------
        x : numpy array
            Symbolical input.

        Returns
        -------
        numpy array
            The numerical output of the net.
        """

    @singledispatchmethod
    def __call__(self, x: SymType) -> SymType:
        return self.forward_sym(x)

    @__call__.register  # type: ignore
    def _(self, x: np.ndarray) -> np.ndarray:
        return self.forward_num(x)


# TODO: eval and train
