import numpy as np
import matplotlib.pyplot as plt

# ▼ 非対称行列（引っ張ってねじる）
A = np.array([[2, 2+0.5],
              [2, 1]])

# ▼ 比較用：対称行列
B = np.array([[2, 2],
              [2, 1]])

# ▼ 単位円上のベクトル群を生成
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape: (2, 200)

# ▼ 行列変換を適用
transformed = A @ circle
transformed_syn = B @ circle

# ▼ プロット
plt.figure(figsize=(6, 6))
plt.plot(circle[0], circle[1], label='Unit Circle (original)', color='gray')
plt.plot(transformed[0], transformed[1], label='Transformed by A', color='red')
plt.plot(transformed_syn[0], transformed_syn[1], label='Transformed by B', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.grid(True)
plt.legend()
plt.title("Effect of Non-Symmetric Matrix A on Unit Circle")
plt.show()
