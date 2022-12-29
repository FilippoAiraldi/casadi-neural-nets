import unittest
from typing import TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnn import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class DummyModule(Module):
    def forward_sym(self, x: SymType) -> SymType:
        return (cs.SX if isinstance(x, cs.SX) else cs.MX).zeros(2, 2)

    def forward_np(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        return np.random.rand(2, 2)


class TestModule(unittest.TestCase):
    def test_call__dispatches_sym_and_numpy_correctly(self):
        module = DummyModule()
        for x in [cs.SX.zeros(2, 2), cs.MX.zeros(2, 2), np.zeros((2, 2))]:
            y = module(x)
            self.assertEqual(type(x), type(y))


if __name__ == "__main__":
    unittest.main()
