import unittest

import casadi as cs
import numpy as np
import torch
import torch.nn as tnn
from parameterized import parameterized, parameterized_class

import csnn as cnn
from csnn import set_sym_type


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestNormalization(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_computes_right_value(self, affine: bool):
        set_sym_type(self.sym_type)
        num_features, batch_size = map(int, np.random.randint(5, 20, 2))
        Lc = cnn.BatchNorm1d(num_features, affine=affine)
        Lt = tnn.BatchNorm1d(num_features, affine=affine, eps=0.0, dtype=float)

        Lt.train()  # prime the torch norm layer
        Lt(torch.randn(batch_size, num_features, dtype=float))
        Lt.eval()

        in_num = np.random.randn(batch_size, num_features)
        out_exp = to_numpy(Lt(torch.from_numpy(in_num)))

        in_sym = Lc.sym_type.sym("x", batch_size, num_features)
        syms = [in_sym, Lc.running_mean, Lc.running_std]
        vals = [in_num, to_numpy(Lt.running_mean), to_numpy(Lt.running_var.sqrt())]
        if affine:
            syms.extend([Lc.weight, Lc.bias])
            vals.extend([to_numpy(Lt.weight), to_numpy(Lt.bias)])
        out_act = cs.substitute(Lc(in_sym), cs.vvcat(syms), cs.vvcat(vals))

        self.assertEqual(out_act.shape, out_exp.shape)
        np.testing.assert_allclose(cs.evalf(out_act), out_exp)

    def test_repr(self):
        set_sym_type(self.sym_type)
        m = cnn.BatchNorm1d(5)
        self.assertIn(cnn.BatchNorm1d.__name__, repr(m))


if __name__ == "__main__":
    unittest.main()
