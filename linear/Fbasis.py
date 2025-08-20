import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# サンプル数
N = 16
n = np.arange(N)

# 基底ベクトル
def dft_basis(k, N, n):
    return np.exp(-2j * np.pi * n * k / N)

# 可視化する k を指定
k = 1

# セットアップ
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.grid(True)
ax.set_title(f"DFT基底ベクトル k={k}")

# 単位円
circle = np.exp(1j * np.linspace(0, 2*np.pi, 400))
ax.plot(circle.real, circle.imag, 'k--', alpha=0.5)

# 点群（更新される）
points, = ax.plot([], [], 'o-', lw=2, label='(w_k)_n')
ax.legend()

# アニメーション初期化
def init():
    points.set_data([], [])
    return points,

# アニメーション更新
def update(frame):
    w_k = dft_basis(k, N, np.arange(frame))
    points.set_data(w_k.real, w_k.imag)
    return points,

ani = animation.FuncAnimation(fig, update, frames=N+1, init_func=init,
                              interval=500, blit=True, repeat=True)

plt.show()
