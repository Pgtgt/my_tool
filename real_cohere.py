import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import coherence

samplerate = 44100
blocksize = 1024*8
nfft = 256
noise_std = 0.1  # ノイズの標準偏差

buffer_x = np.zeros(blocksize)
buffer_y = np.zeros(blocksize)

fig, ax = plt.subplots()
f, Cxy = coherence(buffer_x, buffer_y, fs=samplerate, nperseg=nfft)
line, = ax.plot(f, Cxy)
ax.set_ylim([0, 1.2])
ax.set_xlim([0, samplerate // 2])
ax.set_title("Real-time Coherence (x vs x+noise)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Coherence")

def callback(indata, frames, time, status):
    global buffer_x, buffer_y
    if status:
        print(status)
    x = indata[:, 0]
    noise = np.random.normal(loc=0.0, scale=noise_std, size=len(x))
    y = x + noise
    # y =x

    buffer_x = x
    buffer_y = y

def update_plot(frame):
    global buffer_x, buffer_y
    f, Cxy = coherence(buffer_x, buffer_y, fs=samplerate, nperseg=nfft)
    line.set_ydata(Cxy)
    return line,

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