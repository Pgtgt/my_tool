import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_to_3d_transformation(A, v_list, title_A="2D Before", title_B="3D After"):
    # v_list: shape (N, 2), A: shape (3, 2)
    v_arr = np.array(v_list)
    v_mapped = v_arr @ A.T   # shape (N, 3)

    n_vec = len(v_arr)
    cmap = plt.cm.get_cmap('tab10', n_vec)

    fig = plt.figure(figsize=(12, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # --- Left: 2Dベクトル ---
    for i, v in enumerate(v_arr):
        ax2d.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=cmap(i), alpha=0.7)
    ax2d.set_xlim(-1.2, 1.2)
    ax2d.set_ylim(-1.2, 1.2)
    ax2d.set_aspect('equal')
    ax2d.grid(True)
    ax2d.set_title(title_A)

    # --- Right: 3Dベクトル ---
    for i, v in enumerate(v_mapped):
        ax3d.quiver(0, 0, 0, v[0], v[1], v[2], color=cmap(i), alpha=0.7)
    ax3d.set_xlim(-2, 2)
    ax3d.set_ylim(-2, 2)
    ax3d.set_zlim(-2, 2)
    ax3d.set_title(title_B)

    plt.tight_layout()
    plt.show()

# === 例 ===

# 3×2写像行列
A = np.array([[1, 0.5],
              [0, 1],
              [1, -1]])

# 単位円上に20点
angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
v_list = np.stack([np.cos(angles), np.sin(angles)], axis=1)

plot_2d_to_3d_transformation(A, v_list, "変換前（2D単位円）", "変換後（3D写像）")
