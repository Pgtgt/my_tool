import numpy as np

N = 8
x = np.random.rand(N)
F = np.fft.fft(np.eye(N))  # フーリエ行列
X = F @ x  # 行列積で写像！

print("FFTと一致？", np.allclose(X, np.fft.fft(x)))