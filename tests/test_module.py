import unittest
from typing import TypeVar

import casadi as cs
from parameterized import parameterized_class

from csnn import Module

SymType = TypeVar("SymType", cs.SX, cs.MX)


class DummyModule(Module[SymType]):
    def forward(self, x: SymType) -> SymType:
        return (cs.SX if isinstance(x, cs.SX) else cs.MX).zeros(2, 2)


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestModule(unittest.TestCase):
    def test_register_parameter__raises__with_used_name(self):
        module = DummyModule(self.sym_type)
        module.register_parameter("ciao", cs.SX.zeros(2, 2))
        with self.assertRaisesRegex(KeyError, "Parameter ciao already exists."):
            module.register_parameter("ciao", cs.SX.zeros(2, 2))

    def test_add_module__raises__with_used_name(self):
        module = DummyModule(self.sym_type)
        module.add_module("ciao", DummyModule(self.sym_type))
        with self.assertRaisesRegex(KeyError, "Child module ciao already exists."):
            module.add_module("ciao", DummyModule(self.sym_type))

    def test_add_module__adds_module_correctly(self):
        module = DummyModule(self.sym_type)
        module.ciao = DummyModule(self.sym_type)
        self.assertIn("ciao", module._modules)

    def test_parameters__returns_all_parameters(self):
        # sourcery skip: extract-duplicate-method
        module = DummyModule(self.sym_type)
        p1 = cs.SX.zeros(2, 1)
        module.register_parameter("p1", p1)
        child_module = DummyModule(self.sym_type)
        p2 = cs.SX.zeros(2, 4)
        child_module.register_parameter("p2", p2)
        module.child = child_module

        P = list(module.parameters())

        self.assertEqual(sum(n == "p1" for n, _ in P), 1)
        self.assertEqual(sum(n == "child.p2" for n, _ in P), 1)

    def test_repr(self):
        module = DummyModule(self.sym_type)
        self.assertIn(module.__class__.__name__, repr(module))


if __name__ == "__main__":
    unittest.main()
