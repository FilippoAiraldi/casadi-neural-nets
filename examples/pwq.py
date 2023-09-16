"""Reproduces the PieceWise Quadratic (PWQ) neural network from [1] used to approximate
the value function of a constrained linear proble. Thus, the output of the net is meant
to be a single scalar.

References
----------
[1] He, K., Shi, S., Boom, T.V.D. and De Schutter, B., 2022. Approximate Dynamic
    Programming for Constrained Linear Systems: A Piecewise Quadratic Approximation
    Approach. arXiv preprint arXiv:2205.10065.
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import csnn


class PWQ(csnn.Module):
    def __init__(self, n_in: int, n_hidden: int) -> None:
        super().__init__()
        self.linear = csnn.Sequential((csnn.Linear(n_in, n_hidden), csnn.ReLU()))
        self.output_weights = self.sym_type.sym("r", n_hidden, 1)

    def forward(self, x: cs.MX) -> cs.MX:
        z = self.linear(x)
        z_squared = z * z
        return z_squared @ self.output_weights


# create the model
n_in = 2
n_hidden = 8
mdl = PWQ(n_in, n_hidden)

# turn it into a function
x = cs.MX.sym("x", n_in, 1)
y = mdl(x.T)
p = dict(mdl.parameters())
F = cs.Function("F", [x, cs.vvcat(p.values())], [y], ["x", "p"], ["y"])

# simulate some weights
np_random = np.random.default_rng(69)
p_num = {k: np_random.normal(size=v.shape) for k, v in p.items()}

# force convexity of nn function
p_num["linear.0.bias"] = -np.abs(p_num["linear.0.bias"])
p_num["output_weights"] = +np.abs(p_num["output_weights"])

# plot function
o = np.linspace(-1000, 1000, 1000)
X1, X2 = np.meshgrid(o, o)
X = np.stack((X1, X2)).reshape(n_in, -1)
Y = F(X, cs.vvcat(p_num.values())).full().reshape(o.size, o.size)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X1, X2, Y, cmap="RdBu_r")
plt.show()
