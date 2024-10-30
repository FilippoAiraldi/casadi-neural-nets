"""Recreates a fully connected input convex neural network (FICNN) as per [1]. See also
https://github.com/locuslab/icnn/blob/762cc04992b6e649b4d82953d0c4a821912b1207/synthetic-cls/icnn.py#L213C7-L213C7.

References
----------
[1] Amos, B., Xu, L. and Kolter, J.Z., 2017, July. Input convex neural networks. In
    International Conference on Machine Learning (pp. 146-155). PMLR.
"""


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import csnn

# create the model
n_in = 2
hidden = [32, 16]
n_out = 1
mdl = csnn.convex.FicNN(n_in, hidden, n_out)

# turn it into a function
x = cs.MX.sym("x", n_in, 1)
y = mdl(x.T)
p = dict(mdl.parameters(skip_none=True))
F = cs.Function("F", [x, cs.vvcat(p.values())], [y], ["x", "p"], ["y"])

# initialize the parameters with this class' specific init_parameters method
p_num = dict(mdl.init_parameters(seed=69))

# create a small optimisation problem to enforce that the function is zero at the origin
# and has the global minimum there
opti = cs.Opti("nlp")
p_sym = {k: opti.variable(*v.shape) for k, v in p.items()}
p_sym_v = cs.vvcat(p_sym.values())
opti.minimize(1)
Fjac = F.factory("Fjac", F.name_in(), ["y", "jac:y:x"])
F0, Fjac0 = Fjac(0.0, p_sym_v)  # 0.0 as the origin, but could be any point
opti.subject_to(F0 == 0.0)
opti.subject_to(Fjac0 == 0.0)
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
