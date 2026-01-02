import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# データ生成
X, t = make_swiss_roll(n_samples=2000, noise=0.5, random_state=0)
# X: (n,3), t: “巻き”方向のパラメータ（色付けに便利）

# 3D scatter
fig = plt.figure()
# ax = fig.add_subplot(111,)
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:,0], X[:,1], X[:,2], c=t, s=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
