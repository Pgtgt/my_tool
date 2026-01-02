#%%

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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

# %%
