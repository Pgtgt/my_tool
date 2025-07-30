import numpy as np
import matplotlib.pyplot as plt

# 係数行列Aと定数ベクトルb
A = np.array([[2, 1],
              [1, 2]])
b = np.array([5, 6])

# 答えaを計算
a = np.linalg.inv(A) @ b

# b = Aa, 逆変換 a = A^-1 b
# ベクトル可視化
fig, ax = plt.subplots(figsize=(6,6))
# 原点からa（解ベクトル）
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='green', label='a_unknow')
# 原点からb（もともとの観測ベクトル）
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='blue', label='b')

# 係数行列Aで「aを変換→bになる」ことを示す
# 逆にbにA^-1をかけるとaになる
# （bからaへ点線で）
# ax.plot([b[0], a[0]], [b[1], a[1]], 'r--', lw=2, label='image of inverse transformation')
# ax.quiver(a[0], a[1], b[0]-a[0], b[1]-a[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Aによる写像', width=0.012)
ax.quiver(b[0], b[1], a[0]-b[0], a[1]-b[1], angles='xy', scale_units='xy', scale=1, color='red', label='A⁻¹で b→a', width=0.018)


for pt, label in zip([a, b], ['a', 'b']):
    ax.text(pt[0]*1.05, pt[1]*1.05, label, color='k', fontsize=14)

ax.set_xlim(0, max(b[0], a[0]) + 1)
ax.set_ylim(0, max(b[1], a[1]) + 1)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Aの逆行列で「解a＝A⁻¹b」を可視化')
plt.tight_layout()
plt.show()

print(f"解 a = {a}")
