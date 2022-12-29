import unittest

import casadi as cs
import numpy as np
import torch
from parameterized import parameterized_class
from torch.nn import ReLU as nnReLU

from csnn import ReLU as csReLU


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestActivation(unittest.TestCase):
    def test_computes_right_value(self):
        features = 5, 10
        Lcs = csReLU(self.sym_type)
        Lnn = nnReLU()
        in_num = np.random.randn(*features)
        out_exp = torch_to_numpy(Lnn(torch.from_numpy(in_num)))
        in_sym = Lcs.sym_type.sym("x", *in_num.shape)
        out_act = cs.substitute(Lcs(in_sym), in_sym, in_num)
        self.assertEqual(out_act.shape, out_exp.shape)
        np.testing.assert_allclose(cs.evalf(out_act), out_exp)


if __name__ == "__main__":
    unittest.main()
