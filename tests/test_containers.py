import unittest
from typing import TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from csnn import Module, Sequential

SymType = TypeVar("SymType", cs.SX, cs.MX)


class DummyModule(Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.p = cs.SX.sym(f"p_{name}", 1, 1), np.random.randn()

    def forward_sym(self, input: SymType) -> SymType:
        return input + self.p[0]

    def forward_num(self, input: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        return input + self.p[1]


class TestSequential(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_init__with_dict(self, with_dict: bool):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = Sequential(arg if with_dict else arg.values())
        for i in range(N):
            self.assertIn(f"module{i}" if with_dict else str(i), S._modules)


if __name__ == "__main__":
    unittest.main()
