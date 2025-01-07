import unittest
from itertools import product

import casadi as cs
import numpy as np
from parameterized import parameterized

from csnn import ELU, GELU, SELU, LeakyReLU, Linear, Tanh
from csnn.convex import FicNN, PsdNN, PwqNN
from csnn.feedforward import Mlp


class TestConvex(unittest.TestCase):
    def test__pwqnn__init_parameters(self):
        n_in, n_hidden = np.random.randint(5, 10, size=2)
        mdl = PwqNN(n_in, n_hidden)
        symbols = dict(mdl.parameters())
        for name, value in mdl.init_parameters():
            self.assertIn(name, symbols)
            self.assertTupleEqual(value.shape, symbols[name].shape)

    def test__ficnn__init_parameters(self):
        n_in, n_out = np.random.randint(5, 10, size=2)
        n_hidden = np.random.randint(5, 10, size=3)
        mdl = FicNN(n_in, n_hidden, n_out)
        symbols = dict(mdl.parameters())
        for name, value in mdl.init_parameters():
            self.assertIn(name, symbols)
            self.assertTupleEqual(value.shape, symbols[name].shape)

    @parameterized.expand([("flat",), ("triu",), ("tril",)])
    def test__psdnn__output_shape_and_sparsity(self, out_shape: str):
        n_in, n_out = np.random.randint(5, 10, size=2)
        n_hidden = np.random.randint(5, 10, size=3)
        mdl = PsdNN(n_in, n_hidden, n_out, out_shape)
        x = cs.DM.rand(n_in)
        y: cs.MX = mdl(x.T)
        if out_shape == "flat":
            self.assertTrue(y.is_vector())
            self.assertEqual(y.numel(), n_out * (n_out + 1) // 2)
        elif out_shape == "triu":
            self.assertTrue(y.is_square())
            self.assertTrue(y.is_triu())
            self.assertEqual(y.size1(), n_out)
        else:
            self.assertTrue(y.is_square())
            self.assertTrue(y.is_tril())
            self.assertEqual(y.size1(), n_out)


class TestFeedforward(unittest.TestCase):
    @parameterized.expand(
        product(
            [None, Tanh, [ELU, SELU, LeakyReLU, GELU]],
            [True, False, np.random.choice([True, False], size=4)],
        )
    )
    def test__mlp__init(self, acts, biases):
        features = np.random.randint(5, 10, size=5)
        mdl = Mlp(features, acts, biases)

        actual_acts, actual_biases = [], []
        last_activation = None
        for i, (_, layer) in enumerate(mdl.layers):
            if isinstance(layer, Linear):
                if i > 0:
                    actual_acts.append(last_activation)
                actual_biases.append(layer.bias is not None)
            else:
                last_activation = type(layer)
        actual_acts.append(last_activation)

        self.assertEqual(len(actual_acts) + 1, features.size)
        if acts is None or isinstance(acts, type):
            self.assertTrue(all(b is acts for b in actual_acts))
        else:
            self.assertListEqual(actual_acts, acts)

        self.assertEqual(len(actual_biases) + 1, features.size)
        if isinstance(biases, bool):
            if biases:
                self.assertTrue(all(actual_biases))
            else:
                self.assertFalse(any(actual_biases))
        else:
            self.assertListEqual(actual_biases, biases.tolist())


if __name__ == "__main__":
    unittest.main()
