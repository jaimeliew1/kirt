from kirt import kirt
import numpy as np

from scipy import stats as spstat
from tqdm import trange
import matplotlib.pyplot as plt

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

N = 1
M = 1000
k = 500

# X = np.random.normal(0, 1, [N, M])
# X = np.random.chisquare(1, [N, M])
amin, amax = 1, 5
X = np.zeros([N, M])
for i in range(M):
    X[:, i] = 5 * np.random.weibull(amin + i / M * (amax - amin), N)


filt = kirt.MVSK(X[:, :k])


def make_frame(t):
    for i in range(2):
        filt.update(X[:, make_frame.i])
        make_frame.i += 1
    mean, var, skew, kurt = filt.value[0, :]
    # print(mean - filt.X[0, :].mean())
    # print(var - filt.X[0, :].var())
    # print(skew - spstat.skew(filt.X[0, :]))
    # print(kurt - spstat.kurtosis(filt.X[0, :]))

    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].plot(filt.X[0, :], "kx")
    axes[1].hist(filt.X[0, :], 15, density=True, orientation="horizontal")
    # plt.xlim(-3, 3)
    axes[0].set_ylim(0, 15)
    axes[1].set_xlim(0, 0.5)

    axes[1].axhline(filt.value[0, 0], c="k", lw=1, ls="--")
    # plt.title(f'{filt.value[0,0]:2.2f}')
    return mplfig_to_npimage(fig)


make_frame.i = k


animation = VideoClip(make_frame, duration=10)
animation.write_videofile("rolling_stats.mp4", fps=20)
