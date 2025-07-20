import numpy as np
import matplotlib.pyplot as plt

def plot_2d_transformation_multi_vec(A, v_list, title_A="Before", title_B="After"):
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    # 基底の変換後
    v1 = A @ e1
    v2 = A @ e2

    v_arr = np.array(v_list)
    v_trans = v_arr @ A.T

    n_vec = len(v_arr)
    # カラーマップでn_vec色生成（tab10, tab20, hsv, rainbowなど好きなものを）
    cmap = plt.cm.get_cmap('tab10', n_vec)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left: 元の基底＆任意ベクトル ---
    axs[0].quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='e1' )
    axs[0].quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, color='green', label='e2')
    for i, v in enumerate(v_arr):
        axs[0].quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                      color=cmap(i), width=0.013, label=f'v{i+1}', alpha=0.3)
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-3, 3)
    axs[0].set_aspect('equal')
    axs[0].grid(True)
    axs[0].set_title(title_A)
    # 凡例は一つだけ（重複ラベルを削除）
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')

    # --- Right: 変換後（基底＋任意ベクトル） ---
    # axs[1].quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, color='blue' , alpha=0.3)
    # axs[1].quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.3)
    axs[1].quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='A e1' )
    axs[1].quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green', label='A e2')
    for i, v in enumerate(v_trans):
        axs[1].quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                      color=cmap(i), width=0.013, label=f'A v{i+1}', alpha=0.3)
    axs[1].set_xlim(-3, 3)
    axs[1].set_ylim(-3, 3)
    axs[1].set_aspect('equal')
    axs[1].grid(True)
    axs[1].set_title(title_B)
    axs[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # 行列式の正負
    det = np.linalg.det(A)
    print(f"変換行列 det = {det:.3f} （{'正' if det > 0 else '負'}）")


A2 = np.array([[2, 0],
               [0, 1]])

# ex1. 任意の数の例ベクトル
# v_list = [
#     [1, 1],
#     [2, -1],
#     [-1, 2],
#     [0.5, -2],
#     [-2, -1]
# ]



# ex2.単位円モード。
v_list =[
 [ 1.00000000e+00,  0.00000000e+00],
 [ 9.51056516e-01,  3.09016994e-01],
 [ 8.09016994e-01,  5.87785252e-01],
 [ 5.87785252e-01,  8.09016994e-01],
 [ 3.09016994e-01,  9.51056516e-01],
 [ 6.12323400e-17,  1.00000000e+00],
 [-3.09016994e-01,  9.51056516e-01],
 [-5.87785252e-01,  8.09016994e-01],
 [-8.09016994e-01,  5.87785252e-01],
 [-9.51056516e-01,  3.09016994e-01],
 [-1.00000000e+00,  1.22464680e-16],
 [-9.51056516e-01, -3.09016994e-01],
 [-8.09016994e-01, -5.87785252e-01],
 [-5.87785252e-01, -8.09016994e-01],
 [-3.09016994e-01, -9.51056516e-01],
 [-1.83697020e-16, -1.00000000e+00],
 [ 3.09016994e-01, -9.51056516e-01],
 [ 5.87785252e-01, -8.09016994e-01],
 [ 8.09016994e-01, -5.87785252e-01],
 [ 9.51056516e-01, -3.09016994e-01],]




plot_2d_transformation_multi_vec(A2, v_list, "変換前（標準基底＋例ベクトル）", "変換後（Aによる変換）")

