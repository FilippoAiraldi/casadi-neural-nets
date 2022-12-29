from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import singledispatchmethod
from typing import Any, Dict, Generic, Iterator, Optional, Tuple, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound="Module")
SymType = TypeVar("SymType", cs.SX, cs.MX)


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    return first + "\n" + s


class Module(ABC, Generic[SymType]):
    """Base class for all neural network modules. Your models should also subclass this
    class."""

    def __init__(self) -> None:
        """Initializes the module."""
        self.training: bool = False
        self._sym_parameters: Dict[str, SymType] = OrderedDict()
        self._num_parameters: Dict[str, npt.NDArray[np.double]] = OrderedDict()
        self._modules: Dict[str, "Module"] = OrderedDict()

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

    def children(self) -> Iterator[Tuple[str, "Module"]]:
        """Returns an iterator over immediate children modules.

        Yields
        ------
        Iterator of tuple[str, Module]
            An iterator over tuples of module's names and instances.
        """
        yield from self._modules.items()

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
            for name, module in self.children():
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
            for name, module in self.children():
                yield from module.num_parameters(recurse=True, prefix=f"{prefix}{name}")

    @abstractmethod
    def forward_sym(self, input: SymType) -> SymType:
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
    def forward_num(self, input: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
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

    def train(self: T, mode: bool = True) -> T:
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of particular
        modules for details of their behaviors in training/evaluation mode, if they are
        affected, e.g. `Dropout`, `BatchNorm`, etc.

        Parameters
        ----------
        mode : bool, optional
            Whether to set training mode (`True`) or evaluation mode (`False`). Defaults
            to `True`.

        Returns
        -------
        A reference to itself.
        """
        self.training = mode
        for _, module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        """Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of particular
        modules for details of their behaviors in training/evaluation mode, if they are
        affected, e.g. `Dropout`, `BatchNorm`, etc.

        This is equivalent to `self.train(False)`.

        Returns
        -------
        A reference to itself.
        """
        return self.train(False)

    def extra_repr(self) -> str:
        """Sets the extra representation of the module."""
        return ""

    @singledispatchmethod
    def __call__(self, x: SymType) -> SymType:
        return self.forward_sym(x)

    @__call__.register  # type: ignore
    def _(self, x: np.ndarray) -> np.ndarray:
        return self.forward_num(x)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, (cs.SX, cs.MX)):
            self.register_parameter(name, value)
        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], (cs.SX, cs.MX))
            and isinstance(value[1], np.ndarray)
        ):
            self.register_parameter(name, *value)
        return super().__setattr__(name, value)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        # empty string will be split into list ['']
        if extra_repr := self.extra_repr():
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f"({key}): {mod_str}")
        main_str = f"{self.__class__.__name__}("
        if lines := extra_lines + child_lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
