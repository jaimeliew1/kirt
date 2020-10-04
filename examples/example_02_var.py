from kirt import kirt
import numpy as np
from tqdm import trange

N = 2
M = 1000
k = 100
X = np.random.rand(N, M)

filt = kirt.Var(X[:, :k])
print(filt.value)

for i in range(k, M):
    filt.update(X[:, i])
    print(i, filt.value, filt.direct(filt.X))
