import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import coherence
from matplotlib.widgets import Button


samplerate = 44100
blocksize = 1024
nfft = 512
noise_std = 0.1

buffer_x = np.zeros(blocksize)
buffer_y = np.zeros(blocksize)

fig, ax = plt.subplots()
f, Cxy = coherence(buffer_x, buffer_y, fs=samplerate, nperseg=nfft)

# 初期化：動的コヒーレンス線と、最大記録線
line_current, = ax.plot(f, Cxy, label="Current Coherence")
line_max, = ax.plot(f, np.zeros_like(f), label="Max Coherence", color='red', linestyle='--')

# 最大値を記録するバッファ
max_coherence = np.zeros_like(f)

ax.set_ylim([0, 1.2])
ax.set_xlim([0, samplerate // 2])
ax.set_title("Real-time Coherence (x vs x+noise)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Coherence")
ax.legend()
ax.grid(True)


def callback(indata, frames, time, status):
    global buffer_x, buffer_y
    if status:
        print(status)
    x=indata[:,0]
    noise = np.random.normal(loc=0.0, scale=noise_std, size=len(x))
    y = x + noise
    buffer_x = x
    buffer_y = y

def update_plot(frame):
    global buffer_x, buffer_y, max_coherence
    f, Cxy = coherence(buffer_x, buffer_y, fs=samplerate, nperseg=nfft)
    Cxy = np.nan_to_num(Cxy, nan=0.0)
    Cxy = np.clip(Cxy, 0.0, 1.0)

    print("de")
    print(buffer_x[0])
    print(buffer_y[0])
    print(Cxy[0])
    # print(max_coherence[0])


    # コヒーレンスの最大値を更新
    max_coherence = np.maximum(max_coherence, Cxy)
    # max_coherence = Cxy+0.2

    # 線データ更新
    line_current.set_ydata(Cxy)
    line_max.set_ydata(max_coherence)

    return line_current, line_max

def reset_max(event):
    global max_coherence
    max_coherence[:] = 0  # ゼロリセット
    line_max.set_ydata(max_coherence)
    plt.draw()  # 再描画


# --- ② ボタンを配置する領域を作成 ---
reset_ax = plt.axes([0.8, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
reset_button = Button(reset_ax, 'Reset Max')

# --- ③ ボタンと関数をリンク ---
reset_button.on_clicked(reset_max)


stream = sd.InputStream(
    channels=1,
    samplerate=samplerate,
    blocksize=blocksize,
    dtype='float32',
    callback=callback
)

ani = FuncAnimation(fig, update_plot, interval=100, blit=True)

with stream:
    plt.show()
