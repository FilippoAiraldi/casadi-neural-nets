from typing import Dict, Iterable, Iterator, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnn.module import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Sequential(Module):
    """A sequential container. Modules will be added to it in the order they are passed
    in the constructor. Alternatively, an `OrderedDict` of modules can be passed in.
    The `forward_sym()` and `forward_num()` methods of `Sequential` accept any input and
    forwards it to the first module it contains. It then "chains" outputs to inputs
    sequentially for each subsequent module, finally returning the output of the last
    module."""

    def __init__(self, modules: Union[Dict[str, Module], Iterable[Module]]) -> None:
        """Instianties the sequential module.

        Parameters
        ----------
        modules : dict[str, Module] or iterable of Module
            A dict of names-modules, or an iterable or modules.
        """
        super().__init__()
        if isinstance(modules, dict):
            for name, module in modules.items():
                self.add_module(name, module)
        else:
            for i, module in enumerate(modules):
                self.add_module(str(i), module)

    def forward_sym(self, input: SymType) -> SymType:
        for module in self:
            input = module.forward_sym(input)
        return input

    def forward_num(self, input: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        for module in self:
            input = module.forward_num(input)
        return input

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
