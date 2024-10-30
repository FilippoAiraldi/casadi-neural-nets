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
out_mat_size = 2  # in order to plot the output in 3D
mdl = csnn.convex.PsdNN(n_in, n_hidden, out_mat_size, "tril")

# turn it into a function - it will output a Cholsekly decomposition of a PSD matrix. By
# multiplying the decomposition by its transpose, we get the PSD matrix itself.
x = cs.MX.sym("x", n_in, 1)
L = mdl(x.T)
psd = L @ L.T
p = dict(mdl.parameters(skip_none=True))
F = cs.Function("F", [x, cs.vvcat(p.values())], [psd], ["x", "p"], ["psd"])

# create also a function to compute quadratic forms
z = cs.MX.sym("z", out_mat_size, 1)
M = cs.MX.sym("M", out_mat_size, out_mat_size)
Q = cs.Function("Q", [z, M], [cs.bilin(M, z)], ["z", "M"], ["y"])

# initialize the parameters with this class' specific init_parameters method
p_num = dict(csnn.init_parameters(mdl, seed=np_random))
p_num = cs.vvcat(p_num.values())

# draw a random input and check the correspondig matrix is PSD
x = np_random.normal(size=n_in)
psd = F(x, p_num).toarray()
print("Eig. values of PSD matrix:", np.linalg.eigvals(psd), "(should be >= 0)")

# plot function
o = np.linspace(-3, 3, 100)
x = np.random.randn(n_in)
Z1, Z2 = np.meshgrid(o, o)
Z = np.stack((Z1, Z2)).reshape(out_mat_size, -1)
Y = Q(Z, psd).toarray().reshape(o.size, o.size)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Z1, Z2, Y, cmap="RdBu_r")
ax.set_xlabel(r"$z_1$")
ax.set_ylabel(r"$z_2$")
ax.set_zlabel(r"$y$")
plt.show()
