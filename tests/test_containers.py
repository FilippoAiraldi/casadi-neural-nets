import unittest

import casadi as cs
import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from csnn import Module, Sequential


class DummyModule(Module[cs.SX]):
    def __init__(self, name: str) -> None:
        super().__init__("SX")
        self.p = cs.SX.sym(f"p_{name}", 1, 1), np.random.randn()

    def forward_sym(self, input: cs.SX) -> cs.SX:
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

    def test_forward_num(self):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = Sequential(arg)
        input = 5
        output = S.forward_num(input)
        np.testing.assert_almost_equal(output, input + sum(m.p[1] for _, m in S))

    def test_forward_sym(self):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = Sequential(arg)
        input = cs.SX.sym("x")
        output = S.forward_sym(input)
        self.assertEqual(
            str(output), "((((((((((x+p_0)+p_1)+p_2)+p_3)+p_4)+p_5)+p_6)+p_7)+p_8)+p_9)"
        )


if __name__ == "__main__":
    unittest.main()
