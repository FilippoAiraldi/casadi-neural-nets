"""Recreates a fully connected input convex neural network (FICNN) as per [1]. See also
https://github.com/locuslab/icnn/blob/762cc04992b6e649b4d82953d0c4a821912b1207/synthetic-cls/icnn.py#L213C7-L213C7.

References
----------
[1] Amos, B., Xu, L. and Kolter, J.Z., 2017, July. Input convex neural networks. In
    International Conference on Machine Learning (pp. 146-155). PMLR.
"""


from typing import Callable, Sequence, Tuple, TypeVar

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import csnn

SymType = TypeVar("SymType", cs.SX, cs.MX)


class FicnnLayer(csnn.Module[SymType]):
    """A fully connected input convex neural network (FICNN) layer consisting of two
    linear elements."""

    def __init__(
        self,
        in_features: int,
        prev_hidden_features: int,
        out_features: int,
        act: Callable[[SymType], SymType] = None,
    ) -> None:
        """Creates a layer of a FICNN.

        Parameters
        ----------
        in_features : int
            Feature size of the very first input of the convex net.
        hidden_features : int
            Feature size of the previous hidden layer's output.
        out_features : int
            Feature size of this layer's output.
        act : Callable, optional
            An optional activation function to apply to the output of this layer.
        """
        super().__init__()
        self.y_layer = csnn.Linear(in_features, out_features, bias=True)
        self.z_layer = csnn.Linear(prev_hidden_features, out_features, bias=False)
        self.act = act

    def forward(self, input: Tuple[SymType, SymType]) -> Tuple[SymType, SymType]:
        y, z = input
        z_new = self.y_layer(y) + self.z_layer(z)
        if self.act is not None:
            z_new = self.act(z_new)
        return (y, z_new)


class Ficnn(csnn.Module[SymType]):
    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        act: Callable[[SymType], SymType],
    ) -> None:
        super().__init__()
        if len(hidden_features) < 1:
            raise ValueError("FICNN must have at least one hidden layer")
        self.input_linear = csnn.Linear(in_features, hidden_features[0], True)
        self.input_act = act
        self.hidden_layers = csnn.Sequential(
            FicnnLayer(in_features, hidden_features[i], hidden_features[i + 1], act)
            for i in range(len(hidden_features) - 1)
        )
        self.last_layer = FicnnLayer(
            in_features, hidden_features[-1], out_features, None
        )

    def forward(self, input: SymType) -> SymType:
        y = input
        z1 = self.input_act(self.input_linear(y))
        _, zf_1 = self.hidden_layers((y, z1))  # i.e., z_{f-1}
        return self.last_layer((y, zf_1))[1]


# create the model
n_in = 2
hidden = [32, 16]
n_out = 1
mdl = Ficnn(n_in, hidden, n_out, csnn.SoftPlus())

# turn it into a function
x = cs.MX.sym("x", n_in, 1)
y = mdl(x.T)
p = dict(mdl.parameters(skip_none=True))
F = cs.Function("F", [x, cs.vvcat(p.values())], [y], ["x", "p"], ["y"])

# simulate some weights
np_random = np.random.default_rng(69)
p_num = {k: np_random.normal(size=v.shape) for k, v in p.items()}

# force convexity of nn function
for n, val in p_num.items():
    if n.endswith("z_layer.weight"):
        p_num[n] = np.abs(val)

# create a small optimisation problem to enforce that the function is zero at the origin
# and has the global minimum there
opti = cs.Opti("nlp")
p_sym = {k: opti.variable(*v.shape) for k, v in p.items()}
p_sym_v = cs.vvcat(p_sym.values())
opti.minimize(1)
y = F(x, p_sym_v)
opti.subject_to(cs.substitute(y, x, 0) == 0.0)
opti.subject_to(cs.substitute(cs.jacobian(y, x), x, 0) == 0.0)
for name in p_sym:
    opti.set_initial(p_sym[name], p_num[name])
    if name.endswith("z_layer.weight"):
        opti.subject_to(cs.vec(p_sym[name]) >= 0)
opti.solver("ipopt")
sol = opti.solve()

# plot function
p_num = sol.value(p_sym_v)
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
