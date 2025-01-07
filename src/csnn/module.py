import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from math import prod
from typing import Any, ClassVar, Generic, Optional, TypeVar, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
import casadi as cs

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

    sym_type: ClassVar[Union[type[cs.SX], type[cs.MX]]] = cs.MX

    def __init__(self) -> None:
        self._parameters: dict[str, Optional[SymType]] = OrderedDict()
        self._modules: dict[str, "Module"] = OrderedDict()

    def register_parameter(self, name: str, sym: Optional[SymType]) -> None:
        """Adds a parameter to the module.

        Parameters
        ----------
        name : str
            Name of the parameter.
        sym : SymType
            Symbol of the parameter.

        Raises
        ------
        KeyError
            Raises if `name` is already in use.
        """
        if name in self._parameters:
            raise KeyError(f"Parameter {name} already exists.")
        self._parameters[name] = sym

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

    def children(self) -> Iterator[tuple[str, "Module[SymType]"]]:
        """Returns an iterator over immediate children modules.

        Yields
        ------
        Iterator of tuple[str, Module]
            An iterator over tuples of module's names and instances.
        """
        yield from self._modules.items()

    def parameters(
        self, recurse: bool = True, prefix: str = "", skip_none: bool = False
    ) -> Iterator[tuple[str, SymType]]:
        """Returns an iterator over the module's parameters.

        Parameters
        ----------
        recurse : bool, optional
            If `True`, then yields parameters of this module and all submodules.
            Otherwise, yields only parameters that are direct members of this module. By
            default `True`.
        prefix : str, optional
            Prefix to add in front of this module's name.
        skip_none : bool, optional
            If `True`, then parameters with value `None` are not yielded. By default
            `False`.

        Yields
        ------
        Iterator of tuple[str, casadi.SX or MX or None]
            An iterator over tuples of parameter's names and symbols. If the parameter
            is `None`, and `skip_none=True`, then it is skipped.
        """
        if prefix != "":
            prefix += "."
        for name, par in self._parameters.items():
            if not skip_none or par is not None:
                yield (prefix + name, par)
        if recurse:
            for name, module in self.children():
                yield from module.parameters(True, f"{prefix}{name}", skip_none)

    def apply(self, fn: Callable[["Module[SymType]"], None]) -> Self:
        """Apply `fn` recursively to every submodule (as returned by `.children`) as
        well as self.

        Parameters
        ----------
        fn : callable
            Function to be applied to each submodule.

        Returns
        -------
        Module
            A reference to self.
        """
        for _, module in self.children():
            module.apply(fn)
        fn(self)
        return self

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module and submodules."""
        return sum(prod(p.shape) if p is not None else 0 for _, p in self.parameters())

    @abstractmethod
    def forward(self, *_: SymType, **__: SymType) -> SymType:
        """Forwards symbolically the given input through the neural net.

        Parameters
        ----------
        args, kwargs : SymType
            Symbolical inputs.

        Returns
        -------
        SymType
            The symbolical output of the net.
        """

    def extra_repr(self) -> str:
        """Sets the extra representation of the module."""
        return ""

    def __call__(self, *args: SymType, **kwargs: SymType) -> SymType:
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, (cs.SX, cs.MX)):
            self.register_parameter(name, value)
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
