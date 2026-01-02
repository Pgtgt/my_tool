#%%

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial import distance

# %% data loading & plot
a = pd.read_csv("fig1_1plot.csv") 
x=a["x"].to_numpy().reshape(-1, 1)
y=a["y"].to_numpy().reshape(-1, 1)
plt.plot(x, y)
plt.show()

# %% normal regression
w = np.linalg.inv(x.T @ x)@x.T@y
y_estimate_normalreg = w * x
plt.plot(x, y, label = "row")
plt.plot(x, y_estimate_normalreg, label = "no seppen")
# plt.show()

x_0 = np.ones( len(x)).reshape(-1,1)
x_reg = np.concatenate((x_0, x), axis = 1)
w = np.linalg.inv(x_reg.T @ x_reg)@x_reg.T@y
y_estimate_reg = x_reg @ w
# plt.plot(x, y)
plt.plot(x, y_estimate_reg, label = "seppe")
plt.legend()
plt.show()

# %%  KRR & KRR with norm


D = distance.cdist(x, x, metric="euclidean")
beta = 1
K = np.exp(-beta * D**2)

alpha = np.linalg.inv(K) @ y

x_plot = np.linspace(start=-1.9, stop = 2.0, num = 100)

D_plot = distance.cdist(x, x_plot.reshape(-1,1), metric="euclidean")
K_plot = np.exp(-beta * D_plot**2)

y_estimate_krr = K_plot.T @ alpha

lambda_ = 0.1
alpha_norm = np.linalg.inv(K + lambda_ *np.eye(K.shape[0])) @ y
y_estimate_krr_norm = K_plot.T @ alpha_norm

plt.plot(x_plot, y_estimate_krr, label = "krr")
plt.plot(x_plot, y_estimate_krr_norm, label = "krr_niorm")
plt.scatter(x, y, label = "row")
plt.legend()
plt.show()
# %%
