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

    def forward_num(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        return np.random.rand(2, 2)


class TestModule(unittest.TestCase):
    def test_call__dispatches_sym_and_numpy_correctly(self):
        module = DummyModule()
        for x in [cs.SX.zeros(2, 2), cs.MX.zeros(2, 2), np.zeros((2, 2))]:
            y = module(x)
            self.assertEqual(type(x), type(y))

    def test_register_parameter__raises__with_used_name(self):
        module = DummyModule()
        module.register_parameter("ciao", cs.SX.zeros(2, 2))
        with self.assertRaisesRegex(KeyError, "Parameter ciao already exists."):
            module.register_parameter("ciao", cs.SX.zeros(2, 2))

    def test_register_parameter__instantiates_empty_numerical_array(self):
        module = DummyModule()
        shape = (2, 3)
        module.register_parameter("ciao", cs.SX.zeros(*shape))
        self.assertIn("ciao", module._num_parameters)
        self.assertIsInstance(module._num_parameters["ciao"], np.ndarray)
        self.assertEqual(module._num_parameters["ciao"].shape, shape)

    def test_register_parameter__raises__with_invalid_shape(self):
        module = DummyModule()
        with self.assertRaisesRegex(AssertionError, "Incompatible shapes."):
            module.register_parameter("ciao", cs.SX.zeros(2, 2), np.random.rand(2, 3))

    def test_add_module__raises__with_used_name(self):
        module = DummyModule()
        module.add_module("ciao", DummyModule())
        with self.assertRaisesRegex(KeyError, "Child module ciao already exists."):
            module.add_module("ciao", DummyModule())

    def test_add_module__adds_module_correctly(self):
        module = DummyModule()
        module.add_module("ciao", DummyModule())
        self.assertIn("ciao", module._modules)

    def test_sym_num_parameters__returns_all_parameters(self):
        # sourcery skip: extract-duplicate-method
        module = DummyModule()
        p1 = cs.SX.zeros(2, 1)
        module.register_parameter("p1", p1)
        child_module = DummyModule()
        p2 = cs.SX.zeros(2, 4)
        child_module.register_parameter("p2", p2)
        module.add_module("child", child_module)

        L1 = list(module.sym_parameters())
        L2 = list(module.num_parameters())

        self.assertEqual(sum(n == "p1" for n, _ in L1), 1)
        self.assertEqual(sum(n == "child.p2" for n, _ in L1), 1)
        self.assertEqual(sum(n == "p1" for n, _ in L2), 1)
        self.assertEqual(sum(n == "child.p2" for n, _ in L2), 1)


if __name__ == "__main__":
    unittest.main()
