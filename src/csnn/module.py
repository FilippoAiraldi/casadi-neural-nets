from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Dict, Generic, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Module(ABC, Generic[SymType]):
    def __init__(self) -> None:
        self.training: bool = False
        self._sym_parameters: Dict[str, SymType] = {}
        self._modules: Dict[str, "Module"] = {}

    @abstractmethod
    def forward_sym(self, x: SymType) -> SymType:
        pass

    @abstractmethod
    def forward_np(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        pass

    @singledispatchmethod
    def __call__(self, x: SymType) -> SymType:
        return self.forward_sym(x)

    @__call__.register  # type: ignore
    def _(self, x: np.ndarray) -> np.ndarray:
        return self.forward_np(x)
