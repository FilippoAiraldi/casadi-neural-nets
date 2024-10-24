import unittest

import casadi as cs
import numpy as np
import torch
import torch.nn as tnn
from parameterized import parameterized, parameterized_class

import csnn as cnn
from csnn import set_sym_type


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestLinear(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_computes_right_value(self, bias: bool):
        set_sym_type(self.sym_type)
        in_features, out_features, N = map(int, np.random.randint(5, 20, 3))
        Lcs = cnn.Linear(in_features, out_features, bias=bias)
        Ltorch = tnn.Linear(in_features, out_features, bias=bias, dtype=float)

        in_num = np.random.randn(N, in_features)
        out_exp = torch_to_numpy(Ltorch(torch.from_numpy(in_num)))
        in_sym = Lcs.sym_type.sym("x", N, in_features)
        out_act = cs.substitute(Lcs(in_sym), in_sym, in_num)
        out_act = cs.substitute(out_act, Lcs.weight, torch_to_numpy(Ltorch.weight))
        if bias:
            out_act = cs.substitute(out_act, Lcs.bias.T, torch_to_numpy(Ltorch.bias))

        self.assertEqual(out_act.shape, out_exp.shape)
        np.testing.assert_allclose(cs.evalf(out_act), out_exp)

    def test_repr(self):
        set_sym_type(self.sym_type)
        m = cnn.Linear(5, 10)
        self.assertIn(cnn.Linear.__name__, repr(m))


if __name__ == "__main__":
    unittest.main()
