#%%

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial import distance

from scipy.spatial import distance

# --- kernel definitions ---------------------------------
def rbf_kernel(X, Y, beta=1.0, metric="euclidean"):
    """
    RBF kernel: exp(-beta * ||x - y||^2)
    """
    D = distance.cdist(X, Y, metric=metric)
    return np.exp(-beta * D**2)


def linear_kernel(X, Y):
    """
    Linear kernel: x^T y
    """
    return X @ Y.T


def polynomial_kernel(X, Y, degree=2, coef0=1.0):
    """
    Polynomial kernel: (x^T y + c)^d
    """
    return (X @ Y.T + coef0) ** degree


# --- kernel chooser -------------------------------------
def choose_kernel(kernel_name: str, **kwargs):
    """
    Returns a kernel function with parameters fixed by closure
    """
    if kernel_name.lower() == "rbf":
        beta = kwargs.get("beta", 1.0)
        metric = kwargs.get("metric", "euclidean")
        return lambda X, Y: rbf_kernel(X, Y, beta=beta, metric=metric)

    elif kernel_name.lower() == "linear":
        return lambda X, Y: linear_kernel(X, Y)

    elif kernel_name.lower() == "poly":
        degree = kwargs.get("degree", 2)
        coef0 = kwargs.get("coef0", 1.0)
        return lambda X, Y: polynomial_kernel(X, Y, degree=degree, coef0=coef0)

    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


# %% data loading & plot
a = pd.read_csv("fig1_1plot.csv")
x = a["x"].to_numpy().reshape(-1, 1)
y = a["y"].to_numpy().reshape(-1, 1)

# %% normal regression
w = np.linalg.inv(x.T @ x) @ x.T @ y
y_estimate_normalreg = w * x
plt.plot(x, y, label="row")
plt.plot(x, y_estimate_normalreg, label="no seppen")

x_0 = np.ones(len(x)).reshape(-1, 1)
x_reg = np.concatenate((x_0, x), axis=1)
w = np.linalg.inv(x_reg.T @ x_reg) @ x_reg.T @ y
y_estimate_reg = x_reg @ w
plt.plot(x, y_estimate_reg, label="seppe")
plt.legend()
plt.show()

# %%  KRR & KRR with norm

beta = 1
# K = rbf_kernel_from_cdist(x, x, beta=beta)
kernel_func = choose_kernel("rbf", beta=beta)
K = kernel_func(x, x)



alpha = np.linalg.inv(K) @ y

x_plot = np.linspace(start=-1.9, stop=2.0, num=100)
x_plot_col = x_plot.reshape(-1, 1)

# K_plot = rbf_kernel_from_cdist(x, x_plot_col, beta=beta)
K_plot = kernel_func(x, x_plot_col)


y_estimate_krr = K_plot.T @ alpha

lambda_ = 0.1
alpha_norm = np.linalg.inv(K + lambda_ * np.eye(K.shape[0])) @ y
y_estimate_krr_norm = K_plot.T @ alpha_norm

plt.plot(x_plot, y_estimate_krr, label="krr")
plt.plot(x_plot, y_estimate_krr_norm, label="krr_niorm")
plt.scatter(x, y, label="row")
plt.legend()
plt.show()
# %%
