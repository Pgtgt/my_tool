import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_linked(A, v_list, title_A="Before", title_B="After"):
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    v1 = A @ e1
    v2 = A @ e2
    v3 = A @ e3

    v_arr = np.array(v_list)
    v_trans = v_arr @ A.T
    n_vec = len(v_arr)
    cmap = plt.cm.get_cmap('tab10', n_vec)

    fig = plt.figure(figsize=(14, 6))
    axs = [fig.add_subplot(1, 2, 1, projection='3d'),
           fig.add_subplot(1, 2, 2, projection='3d')]

    # Left: before
    axs[0].quiver(0, 0, 0, e1[0], e1[1], e1[2], color='blue', label='e1')
    axs[0].quiver(0, 0, 0, e2[0], e2[1], e2[2], color='green', label='e2')
    axs[0].quiver(0, 0, 0, e3[0], e3[1], e3[2], color='purple', label='e3')
    for i, v in enumerate(v_arr):
        axs[0].quiver(0, 0, 0, v[0], v[1], v[2], color=cmap(i), alpha=0.3, label=f'v{i+1}')
    axs[0].set_xlim(-2, 2)
    axs[0].set_ylim(-2, 2)
    axs[0].set_zlim(-2, 2)
    axs[0].set_title(title_A)

    # Right: after
    axs[1].quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', label='A e1')
    axs[1].quiver(0, 0, 0, v2[0], v2[1], v2[2], color='green', label='A e2')
    axs[1].quiver(0, 0, 0, v3[0], v3[1], v3[2], color='purple', label='A e3')
    for i, v in enumerate(v_trans):
        axs[1].quiver(0, 0, 0, v[0], v[1], v[2], color=cmap(i), alpha=0.7, label=f'A v{i+1}')
    axs[1].set_xlim(-2, 2)
    axs[1].set_ylim(-2, 2)
    axs[1].set_zlim(-2, 2)
    axs[1].set_title(title_B)

    plt.tight_layout()

    # --- ここから「視点同期」 ---
    # ax0の操作をax1に伝搬
    def on_move(event):
        for ax in axs:
            ax.view_init(elev=axs[0].elev, azim=axs[0].azim)
            ax.set_proj_type('persp')  # 遠近投影を明示（matplotlib3.4以降不要）

        fig.canvas.draw_idle()

    # 片方を動かしたときに同期（ボタン離したとき）
    fig.canvas.mpl_connect('button_release_event', on_move)
    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()

    det = np.linalg.det(A)
    print(f"変換行列 det = {det:.3f} （{'正' if det > 0 else '負'}）")

# === 使い方例 ===
A3 = np.array([[1, 0, 0],
               [0, 2, 1],
               [0, 0, 1]])
# 単位球面点
phi = np.linspace(0, np.pi, 6)
theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
v_list = []
for p in phi:
    for t in theta:
        v_list.append([np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)])
v_list = np.array(v_list)

plot_3d_linked(A3, v_list)
