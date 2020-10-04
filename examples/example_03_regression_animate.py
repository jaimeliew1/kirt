from kirt import kirt
import numpy as np

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import matplotlib.pyplot as plt

N = 2
M = 10000
k = 100
X = np.random.rand(N, M)
Y = np.random.rand(N, M)

ang_vel = 0.001  # rad / step
ang = 0

r = np.random.normal(0, 1, M)
X = np.zeros(M)
Y = np.zeros(M)
for i in range(M):
    X[i] = r[i] * np.cos(np.sin(ang)) + 0.2 * np.random.normal(0, 1)
    Y[i] = r[i] * np.sin(np.sin(ang)) + 0.2 * np.random.normal(0, 1)
    ang += ang_vel


filt = kirt.Regression(X[:k].reshape([1, -1]), Y[:k].reshape([1, -1]))
print(filt.value)
#


def make_frame(t):
    for i in range(2):
        filt.update(X[make_frame.i], Y[make_frame.i])
        make_frame.i += 1
    fig = plt.figure()
    plt.scatter(filt.X, filt.Y, s=1, c="k", marker="x")
    xs = np.linspace(-3, 3)
    ys = xs * filt.value[0, 0]
    plt.plot(xs, ys, "--r", lw=1)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(f"{filt.value[0,0]:2.2f}")
    return mplfig_to_npimage(fig)


make_frame.i = k


animation = VideoClip(make_frame, duration=20)
animation.write_videofile("asdf.mp4", fps=60)
