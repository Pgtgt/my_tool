
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

# 3次元正規分布の設定
mean = np.zeros(3)
# cov = np.eye(3)  # 単位行列 = 標準正規分布 × 3
cov =np.array([
    [1,0,0],
    [0,1,0],
    [0,0,0]
])

# サンプル生成
n_samples = 1000
X = multivariate_normal(mean, cov, size=n_samples)

# 可視化：3D散布図
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, s=10)
ax.set_title("3D Multivariate Normal (mean=0, cov=I)")
ax.set_xlabel("X₁")
ax.set_ylabel("X₂")
ax.set_zlabel("X₃")
plt.tight_layout()
plt.show()
