import numpy as np
import matplotlib.pyplot as plt

def plot_2d(A, ax, title):
    # 基底ベクトル
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    v1 = A @ e1
    v2 = A @ e2

    ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='e1' ,alpha=0.5)
    ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, color='green', label='e2', alpha=0.5)
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label='A e1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='orange', label='A e2')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    ax.legend()

    det = np.linalg.det(A)
    print(f"{title} det = {det:.3f} ( {'正' if det > 0 else '負'} )")

def plot_3d(A):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    e = np.eye(3)
    v1 = A @ e[:, 0]
    v2 = A @ e[:, 1]
    v3 = A @ e[:, 2]

    # 元の基底（青）
    ax1.quiver(0, 0, 0, e[0,0], e[1,0], e[2,0], color='blue', label='e1')
    ax1.quiver(0, 0, 0, e[0,1], e[1,1], e[2,1], color='green', label='e2')
    ax1.quiver(0, 0, 0, e[0,2], e[1,2], e[2,2], color='purple', label='e3')

    # 変換後ベクトル（赤）
    ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='red', label='A e1')
    ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='orange', label='A e2')
    ax1.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='brown', label='A e3')

    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.set_zlim([-2, 2])
    ax1.set_title("Original Matrix A")
    ax1.legend()

    det = np.linalg.det(A)
    print(f"Original A det = {det:.3f} ( {'正' if det > 0 else '負'} )")

    # 符号反転行列
    A_neg = -A

    v1n = A_neg @ e[:, 0]
    v2n = A_neg @ e[:, 1]
    v3n = A_neg @ e[:, 2]

    # 同様にプロット
    ax2.quiver(0, 0, 0, e[0,0], e[1,0], e[2,0], color='blue', label='e1')
    ax2.quiver(0, 0, 0, e[0,1], e[1,1], e[2,1], color='green', label='e2')
    ax2.quiver(0, 0, 0, e[0,2], e[1,2], e[2,2], color='purple', label='e3')

    ax2.quiver(0, 0, 0, v1n[0], v1n[1], v1n[2], color='red', label='-A e1')
    ax2.quiver(0, 0, 0, v2n[0], v2n[1], v2n[2], color='orange', label='-A e2')
    ax2.quiver(0, 0, 0, v3n[0], v3n[1], v3n[2], color='brown', label='-A e3')

    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-2, 2])
    ax2.set_zlim([-2, 2])
    ax2.set_title("Negated Matrix -A")
    ax2.legend()

    det_neg = np.linalg.det(A_neg)
    print(f"Negated -A det = {det_neg:.3f} ( {'正' if det_neg > 0 else '負'} )")

    plt.show()

# === メイン処理 ===

# 2次元例（偶数次元）
A2 = np.array([[2, 1],
               [1, 1]])

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_2d(A2, axs[0], "2D Matrix A")
plot_2d(-A2, axs[1], "2D Matrix -A")
plt.show()

# 3次元例（奇数次元）
A3 = np.array([[1, 0, 0],
               [0, 2, 1],
               [0, 0, 3]])

plot_3d(A3)
