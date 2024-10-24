import unittest
from itertools import product

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
class TestRNNCell(unittest.TestCase):
    @parameterized.expand(product([False, True], ["tanh", "relu"]))
    def test_computes_right_value(self, bias: bool, nonlinearity: str):
        set_sym_type(self.sym_type)
        input_size, hidden_size, N = map(int, np.random.randint(5, 20, 3))
        m_cs = cnn.RNNCell(input_size, hidden_size, bias, nonlinearity)
        m_nn = tnn.RNNCell(input_size, hidden_size, bias, nonlinearity, dtype=float)

        in_num = np.random.randn(N, input_size)
        in_state = np.random.randn(N, hidden_size)
        out_exp = torch_to_numpy(
            m_nn(torch.from_numpy(in_num), torch.from_numpy(in_state))
        )

        out_act = m_cs(in_num, in_state)
        weights = ["weight_ih", "weight_hh"]
        if bias:
            weights += ["bias_ih", "bias_hh"]
        for w in weights:
            sym = getattr(m_cs, w)
            num = torch_to_numpy(getattr(m_nn, w))
            out_act = cs.substitute(out_act, cs.vec(sym), cs.vec(num))
        out_act = cs.evalf(out_act).toarray()

        self.assertEqual(out_act.shape, out_exp.shape)
        np.testing.assert_allclose(out_act, out_exp)

    def test_repr(self):
        set_sym_type(self.sym_type)
        m = cnn.RNNCell(5, 10)
        self.assertIn(cnn.RNNCell.__name__, repr(m))


if __name__ == "__main__":
    unittest.main()