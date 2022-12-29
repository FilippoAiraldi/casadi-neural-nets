from typing import Dict, Iterable, Iterator, Tuple, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnn.module import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Sequential(Module[SymType]):
    """A sequential container. Modules will be added to it in the order they are passed
    in the constructor. Alternatively, an `OrderedDict` of modules can be passed in.
    The `forward_sym()` and `forward_num()` methods of `Sequential` accept any input and
    forwards it to the first module it contains. It then "chains" outputs to inputs
    sequentially for each subsequent module, finally returning the output of the last
    module."""

    def __init__(
        self, modules: Union[Dict[str, Module[SymType]], Iterable[Module[SymType]]]
    ) -> None:
        """Instianties the sequential module.

        Parameters
        ----------
        modules : dict[str, Module] or iterable of Module
            A dict of names-modules, or an iterable or modules.
        """
        names_and_modules: Iterator[Tuple[str, Module[SymType]]] = (
            iter(modules.items())
            if isinstance(modules, dict)
            else map(lambda o: (str(o[0]), o[1]), enumerate(modules))
        )
        first_name, first_module = next(names_and_modules)
        super().__init__(first_module.sym_type.__name__)

        self.add_module(first_name, first_module)
        for name, module in names_and_modules:
            self.add_module(name, module)

    def forward_sym(self, input: SymType) -> SymType:
        for module in self:
            input = module.forward_sym(input)
        return input

    def forward_num(self, input: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        for module in self:
            input = module.forward_num(input)
        return input

    def __iter__(self) -> Iterator[Module[SymType]]:
        return iter(self._modules.values())
