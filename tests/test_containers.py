import unittest

import casadi as cs
import numpy as np
import torch
from parameterized import parameterized
from torch.nn import Linear as nnLinear
from torch.nn import ReLU as nnReLU
from torch.nn import Sequential as nnSequential

from csnn import Linear as csLinear
from csnn import Module as csModule
from csnn import ReLU as csReLU
from csnn import Sequential as csSequential


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


class DummyModule(csModule[cs.SX]):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.p = cs.SX.sym(f"p_{name}", 1, 1)

    def forward(self, input: cs.SX) -> cs.SX:
        return input + self.p


class TestSequential(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_init__with_dict(self, with_dict: bool):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = csSequential(arg if with_dict else arg.values())
        for i in range(N):
            self.assertIn(f"module{i}" if with_dict else str(i), S._modules)

    def test_forward(self):
        N = 10
        arg = {f"module{i}": DummyModule(str(i)) for i in range(N)}
        S = csSequential(arg)
        input = cs.SX.sym("x")
        output = S(input)
        self.assertEqual(
            str(output), "((((((((((x+p_0)+p_1)+p_2)+p_3)+p_4)+p_5)+p_6)+p_7)+p_8)+p_9)"
        )

    def test_with_linear(self):
        Ntorch = nnSequential(
            nnLinear(10, 5, dtype=float),
            nnReLU(),
            nnLinear(5, 1, dtype=float),
            nnReLU(),
        )
        Ncs = csSequential((csLinear(10, 5), csReLU(), csLinear(5, 1), csReLU()))
        in_num = np.random.randn(3, 10)
        out_exp = torch_to_numpy(Ntorch(torch.from_numpy(in_num)))
        in_sym = Ncs.sym_type.sym("x", 3, 10)
        out_act = cs.substitute(Ncs(in_sym), in_sym, in_num)
        pars_cs = dict(Ncs.parameters())
        pars_torch = dict(Ntorch.named_parameters())
        for n in pars_cs:
            out_act = cs.substitute(
                out_act, pars_cs[n], np.atleast_2d(torch_to_numpy(pars_torch[n]))
            )
        self.assertEqual(out_act.shape, out_exp.shape)
        np.testing.assert_allclose(cs.evalf(out_act), out_exp)
        self.assertIn(csSequential.__name__, repr(Ncs))

    def test_indexing(self):
        layers = [csLinear(10, 5), csReLU(), csLinear(5, 1), csReLU()]
        sequential = csSequential(layers)
        for i in range(len(layers)):
            self.assertIs(layers[i], sequential[i])
        self.assertListEqual(layers[1:3], list(sequential[1:3]._modules.values()))
        self.assertListEqual(layers[2:4], list(sequential[2:4]._modules.values()))


if __name__ == "__main__":
    unittest.main()
