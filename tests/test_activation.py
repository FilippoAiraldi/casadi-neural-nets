import unittest
from typing import Callable

import casadi as cs
import numpy as np
import torch
from parameterized import parameterized_class
from torch.nn import ReLU as nnReLU
from torch.nn import Sigmoid as nnSigmoid
from torch.nn import Softplus as nnSoftPlus

from csnn import Module
from csnn import ReLU as csReLU
from csnn import Sigmoid as csSigmoid
from csnn import SoftPlus as csSoftPlus
from csnn import set_sym_type


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestActivation(unittest.TestCase):
    def _test_activation(self, cs_act: Callable, torch_act: Callable):
        set_sym_type(self.sym_type)
        features = np.random.randint(low=1, high=10, size=2)

        Lnn = torch_act()
        in_num = np.random.randn(*features)
        in_num = np.linspace(-4, 4, 1000).reshape(-1, 1)
        out_expected = torch_to_numpy(Lnn(torch.from_numpy(in_num)))

        Lcs: Module = cs_act()
        in_sym = Lcs.sym_type.sym("x", *in_num.shape)
        out_actual = cs.substitute(Lcs(in_sym), in_sym, in_num)

        self.assertEqual(out_actual.shape, out_expected.shape)
        np.testing.assert_allclose(cs.evalf(out_actual), out_expected)

    def test_relu__computes_right_value(self):
        self._test_activation(csReLU, nnReLU)

    def test_sigmoid__computes_right_value(self):
        self._test_activation(csSigmoid, nnSigmoid)

    def test_softplus__computes_right_value(self):
        self._test_activation(csSoftPlus, nnSoftPlus)


if __name__ == "__main__":
    unittest.main()
