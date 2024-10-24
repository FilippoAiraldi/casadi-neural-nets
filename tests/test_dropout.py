import unittest

import casadi as cs
import numpy as np
import torch
from parameterized import parameterized

from csnn import Dropout, Dropout1d


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


class TestDropout(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_computes_right_value(self, training: bool):
        p = 0.75
        m = Dropout(p, training)
        input = cs.DM(torch_to_numpy(torch.rand(20, 16)))
        sum_before = float(cs.sum1(cs.sum2(input)))
        output = m(input)
        sum_after = float(cs.sum1(cs.sum2(output)))

        if training:
            self.assertLessEqual(sum_after * (1 - p), sum_before)
        else:
            self.assertEqual(sum_after, sum_before)

    def test_repr(self):
        m = Dropout(5, 10)
        self.assertIn(Dropout.__name__, repr(m))


class TestDropout1d(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_computes_right_value(self, training: bool):
        p = 0.75
        m = Dropout1d(p, training)
        input = cs.DM(torch_to_numpy(torch.rand(20, 16) + 0.1))
        sum_before = float(cs.sum1(cs.sum2(input)))
        output = m(input)
        sum_after = float(cs.sum1(cs.sum2(output)))

        if training:
            self.assertLessEqual(sum_after * (1 - p), sum_before)
            # check that whole channels are zeros
            for i in range(input.shape[0]):
                if input[i, 0] == 0.0:
                    self.assertEqual(cs.sum2(output[i, :]), 0)
        else:
            self.assertEqual(sum_after, sum_before)

    def test_repr(self):
        m = Dropout1d(5, 10)
        self.assertIn(Dropout1d.__name__, repr(m))


if __name__ == "__main__":
    unittest.main()
