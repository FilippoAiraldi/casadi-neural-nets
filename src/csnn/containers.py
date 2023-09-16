from typing import Dict, Iterable, Iterator, Tuple, TypeVar, Union

import casadi as cs

from csnn.module import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Sequential(Module[SymType]):
    """A sequential container. Modules will be added to it in the order they are passed
    in the constructor. Alternatively, an `OrderedDict` of modules can be passed in.
    The `forward` method of `Sequential` accepts any input and forwards it to the first
    module it contains. It then "chains" outputs to inputs sequentially for each
    subsequent module, finally returning the output of the last module."""

    def __init__(
        self, modules: Union[Dict[str, Module[SymType]], Iterable[Module[SymType]]]
    ) -> None:
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

    def forward(self, input: SymType) -> SymType:
        for _, module in self:
            input = module.forward(input)
        return input

    def __iter__(self) -> Iterator[Tuple[str, Module[SymType]]]:
        return iter(self._modules.items())
