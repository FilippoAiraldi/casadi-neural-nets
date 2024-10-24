import unittest

import casadi as cs
import numpy as np
import torch
from parameterized import parameterized, parameterized_class
from torch.nn import Linear as nnLinear

from csnn import Linear as csLinear
from csnn import set_sym_type


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestLinear(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_computes_right_value(self, bias: bool):
        set_sym_type(self.sym_type)
        features = 5, 10
        Lcs = csLinear(*features, bias=bias)
        Ltorch = nnLinear(*features, bias=bias, dtype=float)

        N = 12
        in_num = np.random.randn(N, features[0])
        out_exp = torch_to_numpy(Ltorch(torch.from_numpy(in_num)))
        in_sym = Lcs.sym_type.sym("x", N, features[0])
        out_act = cs.substitute(Lcs(in_sym), in_sym, in_num)
        out_act = cs.substitute(out_act, Lcs.weight, torch_to_numpy(Ltorch.weight))
        if bias:
            out_act = cs.substitute(out_act, Lcs.bias.T, torch_to_numpy(Ltorch.bias))

        self.assertEqual(out_act.shape, out_exp.shape)
        np.testing.assert_allclose(cs.evalf(out_act), out_exp)

    def test_repr(self):
        set_sym_type(self.sym_type)
        L = csLinear(5, 10)
        self.assertIn(csLinear.__name__, repr(L))


if __name__ == "__main__":
    unittest.main()
