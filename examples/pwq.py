"""Reproduces the PieceWise Quadratic (PWQ) neural network from [1] used to approximate
the value function of a constrained linear problem. Thus, the output of the net is meant
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

# create the model
n_in = 2
n_hidden = 32
mdl = csnn.convex.PwqNN(n_in, n_hidden)

# turn it into a function
x = cs.MX.sym("x", n_in, 1)
y = mdl(x.T)
p = dict(mdl.parameters(skip_none=True))
F = cs.Function("F", [x, cs.vvcat(p.values())], [y], ["x", "p"], ["y"])

# initialize the parameters with this class' specific init_parameters method
p_num = dict(mdl.init_parameters(seed=69))

# plot function
x = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x, x)
X = np.stack((X1, X2)).reshape(n_in, -1)
p_num = cs.vvcat(p_num.values())
Y = F(X, p_num).full().reshape(x.size, x.size)

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
