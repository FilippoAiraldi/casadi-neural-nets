import unittest

import numpy as np
import torch.nn as tnn
from parameterized import parameterized

import csnn as cnn

# we cannot test that the initializations in this package and pytorch are the same, but
# we can compute their distribution's moments and check that they are somewhat close


class TestInit(unittest.TestCase):
    def _test_stats(self, mdl_cnn: cnn.Module, mdl_tnn: tnn.Module):
        pars_cnn = cnn.init_parameters(mdl_cnn)
        stats_cnn = {n: (p.mean(), p.std()) for n, p in pars_cnn}
        stats_tnn = {
            n: (p.mean().item(), p.std().item()) for n, p in mdl_tnn.named_parameters()
        }
        self.assertSetEqual(set(stats_cnn.keys()), set(stats_tnn.keys()))
        for n in stats_cnn:
            mean1, std1 = stats_cnn[n]
            mean2, std2 = stats_tnn[n]
            np.testing.assert_allclose(mean1, mean2, rtol=1e-1, atol=1e-1)
            np.testing.assert_allclose(std1, std2, rtol=1e-1, atol=1e-1)

    @parameterized.expand([(False,), (True,)])
    def test_init__linear(self, bias: bool):
        sizes = (1000, 100)
        mdl_cnn = cnn.Linear(*sizes, bias=bias)
        mdl_tnn = tnn.Linear(*sizes, bias=bias, dtype=float)
        self._test_stats(mdl_cnn, mdl_tnn)

    @parameterized.expand([(False,), (True,)])
    def test_init__rnn_cell(self, bias: bool):
        sizes = (1000, 100)
        mdl_cnn = cnn.RNNCell(*sizes, bias=bias)
        mdl_tnn = tnn.RNNCell(*sizes, bias=bias, dtype=float)
        self._test_stats(mdl_cnn, mdl_tnn)

    @parameterized.expand([(False,), (True,)])
    def test_init__rnn(self, bias: bool):
        sizes = (100, 100, 10)
        mdl_cnn = cnn.RNN(*sizes, bias=bias)
        mdl_tnn = tnn.RNN(*sizes, bias=bias, dtype=float)
        self._test_stats(mdl_cnn, mdl_tnn)


if __name__ == "__main__":
    unittest.main()
