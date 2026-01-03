#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from icecream import ic
from scipy.spatial import distance

# データ生成
X, t = make_swiss_roll(n_samples=2000, noise=0.5, random_state=0)
# X: (n,3), t: “巻き”方向のパラメータ（色付けに便利）


# X[:,1]=X[:,1]*10 # TODO delete
# 3D scatter
fig = plt.figure()
# ax = fig.add_subplot(111,)
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:,0], X[:,1], X[:,2], c=t, s=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

#%%
ic(X[:,0].mean())
ic(X[:,1].mean())
ic(X[:,2].mean())

ic(X[:,0].var()**0.5)
ic(X[:,1].var()**0.5)
ic(X[:,2].var()**0.5)

# %% PCA
X_centered = X - X.mean(axis=0, keepdims=True)
n = X_centered.shape[0]
Sigma = (X_centered.T @ X_centered) / n
ic(Sigma.shape)  # (3,3)
eigvals, eigvecs = np.linalg.eigh(Sigma)
idx = np.argsort(eigvals)[::-1]   # 降順
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
V2 = eigvecs[:, :2]    # (3,2)
Lambda2 = eigvals[:2]  # (2,)
P = V2 @ V2.T   # (3,3)
X_proj_3d = (P @ X_centered.T).T   # (n,3)
Z = X_centered @ V2   # (n,2)
plt.figure()
plt.scatter(Z[:,0], Z[:,1], s=5)
plt.title("PCA (no kernel)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis("equal")
plt.show()

# 3D scatter
fig = plt.figure()
# ax = fig.add_subplot(111,)
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_proj_3d[:,0], X_proj_3d[:,1], X_proj_3d[:,2], c=t, s=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

# %%


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
