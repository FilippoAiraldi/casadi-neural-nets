"""Reproduces the PieceWise Quadratic (PWQ) neural network from [1] used to approximate
the value function of a constrained linear proble. Thus, the output of the net is meant
to be a single scalar.

References
----------
[1] He, K., Shi, S., Boom, T.V.D. and De Schutter, B., 2022. Approximate Dynamic
    Programming for Constrained Linear Systems: A Piecewise Quadratic Approximation
    Approach. arXiv preprint arXiv:2205.10065.
"""

from typing import Callable, TypeVar

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import csnn

SymType = TypeVar("SymType", cs.SX, cs.MX)


class ElementWiseSquare(csnn.Module[SymType]):
    """Squares the input in an element-wise fashion."""

    def forward(self, x: SymType) -> SymType:
        return x * x


class DotProduct(csnn.Linear[SymType]):
    """Compute the dot product of the input with some weights."""

    def __init__(self, in_features: int) -> None:
        super().__init__(in_features, 1, False)


class Pwq(csnn.Module[SymType]):
    """Piecewise quadratic (PWQ) neural network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act: Callable[[SymType], SymType] = csnn.ReLU(),
    ) -> None:
        """Creates a PWQ model.

        Parameters
        ----------
        in_features : int
            Number of input features.
        hidden_features : int
            Number of features in the single hidden layer.
        act : Callable[[SymType], SymType], optional
            Instance of an activation function class, or a callable that computes an
            activation function. By default, `ReLU` is used.
        """
        super().__init__()
        self.layers = csnn.Sequential(
            (
                csnn.Linear(in_features, hidden_features),
                act,
                ElementWiseSquare(),
                DotProduct(hidden_features),
            )
        )

    def forward(self, x: SymType) -> SymType:
        return self.layers(x)


# create the model
n_in = 2
n_hidden = 32
mdl = Pwq(n_in, n_hidden)

# turn it into a function
x = cs.MX.sym("x", n_in, 1)
y = mdl(x.T)
p = dict(mdl.parameters(skip_none=True))
F = cs.Function("F", [x, cs.vvcat(p.values())], [y], ["x", "p"], ["y"])

# simulate some weights
np_random = np.random.default_rng(69)
p_num = {k: np_random.normal(size=v.shape) for k, v in p.items()}

# force convexity of nn function
p_num["layers.3.weight"] = np.abs(p_num["layers.3.weight"])

# force value at origin to be close to zero
p_num["layers.0.bias"] = -np.abs(p_num["layers.0.bias"])

# plot function
p_num = cs.vvcat(p_num.values())
o = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(o, o)
X = np.stack((X1, X2)).reshape(n_in, -1)
Y = F(X, p_num).full().reshape(o.size, o.size)

print("Value in origin:", F(0, p_num).full().item(), "(should be zero)")
min_idx = np.argmin(Y)
r = np.unravel_index(min_idx, Y.shape)
print(
    "Minimum value:", Y[r], "at", X[:, min_idx], "(should be zero, and close to origin)"
)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X1, X2, Y, cmap="RdBu_r")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")
plt.show()
