import numpy as np
import matplotlib.pyplot as plt

# 2x2行列（対称行列）
A = np.array([[2, 2],
              [1, 4]])

# グリッド作成
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2  # Q(x,y) = x^T A x

# 3Dサーフェス
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# 等高線もz=0平面に投影
contour = ax.contour(X, Y, Z, levels=15, cmap='cool', offset=Z.min(), linestyles='solid', linewidths=1)

# 主軸（固有ベクトル）をxy平面上に赤い矢印で描く
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)
center = np.array([0, 0])
scale = 1.2
for i in range(2):
    eigv = eigvecs[:, i]
    ax.quiver(0, 0, Z.min(), scale*eigv[0], scale*eigv[1], 0,
              color='red', linewidth=2, arrow_length_ratio=0.08)
    ax.text(*(scale*eigv[0], scale*eigv[1], Z.min()), f"eigv{i+1}", color='red', fontsize=12)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Q(x, y)")
ax.set_title("2次形式 $Q(x, y) = [x, y] A [x, y]^T$ の3D可視化")
ax.view_init(elev=30, azim=35)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Q(x, y)')
plt.tight_layout()
plt.show()
