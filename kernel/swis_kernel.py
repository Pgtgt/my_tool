#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from icecream import ic
from scipy.spatial import distance

# データ生成
X, t = make_swiss_roll(n_samples=2000, noise=0.5, random_state=0)
# X: (n,3), t: “巻き”方向のパラメータ（色付けに便利）

# %% tilt

yaw   = np.deg2rad(20)  # z軸まわり
pitch = np.deg2rad(-35) # y軸まわり
roll  = np.deg2rad(10)  # x軸まわり

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
X = X @ R.T


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
sc = plt.scatter(
    Z[:, 0],
    Z[:, 1],
    c=t,          # ← ここ
    cmap="viridis",
    s=8
)
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

# %% Kernel def

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


#%% K PCA.  2D ver + 3Dver


beta = 0.001
# K = rbf_kernel_from_cdist(x, x, beta=beta)
kernel_func = choose_kernel("rbf", beta=beta)
K = kernel_func(X, X)
n = X.shape[0]
I = np.eye(n)
one = np.ones((n, n))
J = I - one / n 
JK = J @ K# centered Gram Mat

eigvals, eigvecs = np.linalg.eigh(JK)
idx = np.argsort(eigvals)[::-1]   # 降順
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
plt.scatter(np.arange(len(eigvals)),eigvals)
plt.ylabel("eig val(K PCA)")
plt.show()

lim_a_num =  2
V2 = eigvecs[:, :lim_a_num]    # (3,2)
Lambda2 = eigvals[:lim_a_num]  # (2,)
P = V2 @ V2.T   # (3,3)
# X_proj_3d = (P @ X_centered.T).T   # (n,3)
Z = JK @ V2   # (n,2)
plt.figure()
plt.scatter(Z[:,0], Z[:,1], s=8, c= t)
plt.title("PCA ( kernel) beta:{}".format(beta))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis("equal")
plt.show()


lim_a_num =  3
V3 = eigvecs[:, :lim_a_num]    # (3,2)
Lambda2 = eigvals[:lim_a_num]  # (2,)
P = V3 @ V3.T   # (3,3)
# X_proj_3d = (P @ X_centered.T).T   # (n,3)
Z = JK @ V3   # (n,2)

fig = plt.figure()
# ax = fig.add_subplot(111,)
ax = fig.add_subplot(111, projection="3d")

ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=t, s=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


# %%
