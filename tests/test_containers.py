import unittest

import casadi as cs
from parameterized import parameterized

from csnn import Module, Sequential


class DummyModule(Module[cs.SX]):
    def __init__(self, name: str) -> None:
        super().__init__("SX")
        self.p = cs.SX.sym(f"p_{name}", 1, 1)

    def forward(self, input: cs.SX) -> cs.SX:
        return input + self.p


class TestSequential(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_init__with_dict(self, with_dict: bool):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = Sequential(arg if with_dict else arg.values())
        for i in range(N):
            self.assertIn(f"module{i}" if with_dict else str(i), S._modules)

    def test_forward(self):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = Sequential(arg)
        input = cs.SX.sym("x")
        output = S(input)
        self.assertEqual(
            str(output), "((((((((((x+p_0)+p_1)+p_2)+p_3)+p_4)+p_5)+p_6)+p_7)+p_8)+p_9)"
        )


if __name__ == "__main__":
    unittest.main()
