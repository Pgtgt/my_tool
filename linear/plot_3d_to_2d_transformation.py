import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_to_2d_transformation(A, v_list, title_A="3D Before", title_B="2D After"):
    # v_list: shape (N, 3), A: shape (2, 3)
    v_arr = np.array(v_list)
    v_mapped = v_arr @ A.T   # shape (N, 2)

    n_vec = len(v_arr)
    cmap = plt.cm.get_cmap('tab10', n_vec)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    # --- Left: 3Dベクトル ---
    for i, v in enumerate(v_arr):
        ax3d.quiver(0, 0, 0, v[0], v[1], v[2], color=cmap(i), alpha=0.7)
    ax3d.set_xlim(-1.2, 1.2)
    ax3d.set_ylim(-1.2, 1.2)
    ax3d.set_zlim(-1.2, 1.2)
    ax3d.set_title(title_A)

    # --- Right: 2Dに写像したベクトル ---
    for i, v in enumerate(v_mapped):
        ax2d.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=cmap(i), alpha=0.7)
    ax2d.set_xlim(-2, 2)
    ax2d.set_ylim(-2, 2)
    ax2d.set_aspect('equal')
    ax2d.grid(True)
    ax2d.set_title(title_B)

    plt.tight_layout()
    plt.show()

# === 例 ===

# 2×3写像行列（適当に設計）
A = np.array([[1, 0.5, 0.2],
              [-0.3, 1, -0.7]])

# 3次元単位球面上に点（phi: 0〜π, theta: 0〜2π）
phi = np.linspace(0, np.pi, 5)
theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
v_list = []
for p in phi:
    for t in theta:
        v_list.append([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)])
v_list = np.array(v_list)

plot_3d_to_2d_transformation(A, v_list, "変換前（3D単位球）", "変換後（2D写像）")
