from kirt import kirt
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

N = 1000
M = 2100
k = 2000

X = np.random.rand(N, M)
Y = np.random.rand(N, M)
filt = kirt.Regression(X[:, :k], Y[:, :k])

for i in trange(k, M, desc="real-time"):
    filt.update(X[:, i], Y[:, i])

X = np.random.rand(N, M)
Y = np.random.rand(N, M)
filt = kirt.Regression(X[:, :k], Y[:, :k])

for i in trange(k, M, desc="direct"):
    filt.update(X[:, i], Y[:, i])
    filt.direct(filt.X, filt.Y)
