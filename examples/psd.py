"""This example shows the PsdNN class in action. Given an input, it yields the Cholesky
decomposition of a positive semi-definite matrix (PSD).
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import csnn

# create the model
np_random = np.random.default_rng(69)
n_in = 10
n_hidden = [32, 16, 8]
out_size = 2  # in order to plot the output in 3D
mdl = csnn.convex.PsdNN(n_in, n_hidden, out_size, "tril")

# turn it into a function - it will output a quadratic form based on the Cholsekly
# decomposition of a PSD matrix and a reference point.
z = cs.MX.sym("z", n_in, 1)  # context of the NN to predict the quadratic form
x = cs.MX.sym("x", out_size, 1)  # space where the quadratic form lies
y = mdl.quadform(x.T, z.T)
# L, ref = mdl(z.T)  # if done manually
# y = cs.bilin(L @ L.T + mdl._eps, x - ref)
p = cs.veccat(*(p for _, p in mdl.parameters(skip_none=True)))
QF = cs.Function("QF", [x, z, p], [y], ["x", "z", "p"], ["y"])

# initialize the parameters with this class' specific init_parameters method
p_num = dict(csnn.init_parameters(mdl, seed=np_random))
p_num = cs.vvcat(p_num.values())

# draw a random context point and check the correspondig matrix is PSD
z_num = np_random.normal(size=(n_in, 1))
L, _ = mdl(z_num.T)
psd = cs.evalf(cs.substitute(L @ L.T + mdl._eps, p, p_num))
print("Eig. values of PSD matrix:", np.linalg.eigvals(psd), "(should be >= 0)")

# plot the corresponding quadratic form in the output space
o = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(o, o)
X = np.stack((X1, X2)).reshape(out_size, -1)
Y = QF(X, z_num, p_num).toarray().reshape(o.size, o.size)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X1, X2, Y, cmap="RdBu_r")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")
plt.show()
