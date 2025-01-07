import unittest
from typing import Callable

import casadi as cs
import numpy as np
import torch
import torch.nn as tnn
from parameterized import parameterized_class

import csnn as cnn


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestActivation(unittest.TestCase):
    def _test_activation(self, cs_act: Callable, torch_act: Callable):
        cnn.set_sym_type(self.sym_type)
        features = np.random.randint(low=1, high=10, size=2)
        in_num = np.random.randn(*features) * 10.0

        act_torch = torch_act()
        out_expected = torch_to_numpy(act_torch(torch.from_numpy(in_num)))

        act_csnn = cs_act()
        in_sym = act_csnn.sym_type.sym("x", *in_num.shape)
        out_actual = cs.substitute(act_csnn(in_sym), in_sym, in_num)
        out_actual = cs.evalf(out_actual).toarray().reshape(in_num.shape)

        self.assertEqual(out_actual.shape, out_expected.shape, msg=act_csnn)
        np.testing.assert_allclose(out_actual, out_expected, err_msg=act_csnn)

    def test_gelu(self):
        self._test_activation(cnn.GELU, tnn.GELU)

    def test_leaky_relu(self):
        self._test_activation(cnn.LeakyReLU, tnn.LeakyReLU)

    def test_relu(self):
        self._test_activation(cnn.ReLU, tnn.ReLU)

    def test_elu(self):
        self._test_activation(cnn.ELU, tnn.ELU)

    def test_selu(self):
        self._test_activation(cnn.SELU, tnn.SELU)

    def test_sigmoid(self):
        self._test_activation(cnn.Sigmoid, tnn.Sigmoid)

    def test_softplus(self):
        self._test_activation(cnn.Softplus, tnn.Softplus)

    def test_tanh(self):
        self._test_activation(cnn.Tanh, tnn.Tanh)

    # def test_gelu(self):
    #     self._test_activation(
    #         partial(cnn.GELU, approximate="tanh"), partial(tnn.GELU, approximate="tanh")
    #     )


if __name__ == "__main__":
    unittest.main()
