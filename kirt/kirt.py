import numpy as np
from scipy.special import binom
from scipy import stats as spstat
from scipy import sparse


class Kirt_base(object):
    def __init__(self):
        pass

    def direct(self, X):
        pass

    def update(self, x):
        pass

    @property
    def value(self):
        pass


class Mean(Kirt_base):
    # commonly used in finance (SMA)
    def __init__(self, X):
        self.sum = X.sum(axis=1)
        self.X = X
        self.N, self.M = X.shape

    @property
    def direct(self):
        return self._direct(self.X)

    def _direct(self, X):
        return X.mean(axis=1)

    def update(self, x_in):
        # 0 is the oldest.
        x_out = self.X[:, 0]
        self.sum += x_in - x_out
        self.X = np.roll(self.X, -1)
        self.X[:, -1] = x_in

        return self.value

    @property
    def value(self):
        return self.sum / self.M


class Var(Kirt_base):
    # https://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
    def __init__(self, X):
        self.mean = Mean(X)
        self.sum = np.sum(X - self.mean.value.reshape([-1, 1]) ** 2, axis=1)
        self.X = X
        self.N, self.M = X.shape

    @property
    def direct(self):
        return self._direct(self.X)

    def _direct(self, X):
        return X.var(axis=1)
        # return np.mean(X - X.mean(axis=1).reshape([-1, 1]) ** 2, axis=1)

    def update(self, x_in):
        # 0 is the oldest.
        x_out = self.X[:, 0]

        self.X = np.roll(self.X, -1)
        self.X[:, -1] = x_in

        mean_old = self.mean.value
        mean_new = self.mean.update(x_in)

        self.sum += (x_in - x_out) * (x_out - mean_new + x_in - mean_new)

        return self.value

    @property
    def value(self):
        return self.sum / self.M


class CentralMoment(Kirt_base):
    # https://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    def __init__(self, X, n, first_moment="mean"):
        self.X = X
        self.N, self.M = X.shape
        self.n = n
        self.first_moment = first_moment

        self.moments = self.direct(X)
        # _sum = []
        # for i in range(self.N):
        #     _sum.append(np.vander(X[i, :], n + 1, increasing=True).T.sum(1))
        # self.sum = np.column_stack(_sum).T
        # self.moments = self._calc_moments(self.sum, self.n, self.M, self.first_moment)

    @staticmethod
    def _calc_moments(sums, n, M, first_moment="mean"):
        N = sums.shape[0]
        moment = np.zeros([N, n])
        for i in range(1, n + 1):
            if i == 1 and first_moment == "mean":
                moment[:, 0] = sums[:, 1] / M

                continue
            # Papoulis 1984, p. 146
            for k in range(i + 1):
                moment[:, i - 1] += (
                    binom(i, k) * sums[:, k] / M * (-sums[:, 1] / M) ** (i - k)
                )
        return moment

    @property
    def direct(self):
        return self._direct(self.X)

    def _direct(self, X):
        _sum = []
        for i in range(self.N):
            _sum.append(np.vander(X[i, :], self.n + 1, increasing=True).T.sum(1))
        self.sum = np.column_stack(_sum).T  # TODO this should be in init
        return self._calc_moments(self.sum, self.n, self.M, self.first_moment)

    def update(self, x_in):
        # 0 is the oldest.
        x_out = self.X[:, 0]
        self.X = np.roll(self.X, -1)
        self.X[:, -1] = x_in

        sum_in = np.vander(x_in, self.n + 1, increasing=True)
        sum_out = np.vander(x_out, self.n + 1, increasing=True)

        self.sum += sum_in - sum_out

        self.moments = self._calc_moments(self.sum, self.n, self.M, self.first_moment)
        return self.value

    @property
    def value(self):
        return self.moments


class MVSK(Kirt_base):
    """
    Rolling mean, variance, skewness and excess kurtosis.
    """

    def __init__(self, X):
        self.rolling_moment = CentralMoment(X, n=4)
        self.stats = self._moment_to_stat(self.rolling_moment.value)

    @staticmethod
    def _moment_to_stat(moment):
        stats = np.zeros_like(moment)
        stats[:, 0] = moment[:, 0]  # mean
        stats[:, 1] = moment[:, 1]  # variance
        stats[:, 2] = moment[:, 2] / moment[:, 1] ** 1.5  # skewness
        stats[:, 3] = moment[:, 3] / moment[:, 1] ** 2 - 3  # excess kurtosis

        return stats

    @property
    def direct(self):
        return self._direct(self.X)

    def _direct(self, X):
        rolling_moment = CentralMoment(X, n=4)
        return self._moment_to_stat(rolling_moment.value)

    def update(self, x):
        self.rolling_moment.update(x)
        self.stats = self._moment_to_stat(self.rolling_moment.value)
        return self.stats

    @property
    def value(self):
        return self.stats

    @property
    def X(self):
        return self.rolling_moment.X


class Covariance(Kirt_base):
    pass


class DFT(Kirt_base):
    pass


class Regression(Kirt_base):
    # https://stats.stackexchange.com/questions/266631/what-is-the-difference-between-least-square-and-pseudo-inverse-techniques-for-li
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.A = Y @ np.linalg.pinv(X)
        self.P = np.linalg.inv(X @ X.T)

    @property
    def direct(self):
        return self._direct(self.X, self.Y)

    def _direct(self, X, Y):
        return Y @ np.linalg.pinv(X)

    def update(self, x_in, y_in):
        x_out, y_out = self.X[:, 0], self.Y[:, 0]
        # Update recent w snapshots
        self.X = np.column_stack((self.X[:, 1:], x_in))
        self.Y = np.column_stack((self.Y[:, 1:], y_in))

        # direct rank-2 update
        # define matrices
        U = np.column_stack((x_out, x_in))
        V = np.column_stack((y_out, y_in))

        C = np.diag([-1, 1])
        # compute PkU matrix matrix product beforehand
        PkU = self.P @ U
        # compute AkU matrix matrix product beforehand
        AkU = self.A @ U
        # compute Gamma
        Gamma = np.linalg.inv(np.linalg.inv(C) + U.T @ PkU)
        # update A
        self.A += (V - AkU) @ Gamma @ PkU.T
        # update P
        self.P = self.P - PkU @ Gamma @ PkU.T
        # ensure P is SPD by taking its symmetric part
        self.P = (self.P + self.P.T) / 2

    @property
    def value(self):
        return self.A


class SVD(Kirt_base):
    def __init__(self, X, trunc=None):
        self.X = X
        if trunc is not None:
            self.trunc = min(trunc, X.shape[0])
        else:
            self.trunc = X.shape[0]

        self.U, self.S, self.V = self.direct(X, self.trunc)

    @property
    def direct(self):
        return self._direct(self.X, self.trunc)

    def _direct(self, X, trunc):
        U, S, V = np.linalg.svd(X, full_matrices=False)

        V = V.conj().T

        U_r = U[:, :trunc]
        S_r = S[:trunc]
        V_r = V[:, :trunc]

        return U_r, S_r, V_r

    def update(self, x_in):
        """
        Based on Brand, M. 'Fast low-rank modifications of the thin singular
        value decomposition' (2005)
        """
        x_out = self.X[:, 0]
        # Update recent w snapshots
        self.X = np.column_stack((self.X[:, 1:], x_in))

        r = self.S.shape[0]

        A = sparse.diags(self.S)
        # variable, m, in Eq. (6)
        B = (self.U.T @ x_in).reshape(-1, 1)
        C = np.zeros((1, r))
        # variable, R_a in Eq. (6)
        p = np.linalg.norm((np.eye(len(x_in)) - self.U @ self.U.T) @ (x_in - x_out))
        D = np.array([[p]])

        # Eq. (9)
        K = sparse.bmat([[A, B], [C, D]], format="csr")

        # k=min(K.shape) - 1
        k = self.trunc
        U_, S_, _ = sparse.linalg.svds(K, k=k, return_singular_vectors="u")
        # order singular values in descending order.
        S_ = S_[::-1]

        # U_, S_, _ = self.direct(K.toarray(), trunc=k) # seems to be faster than sparse method for large ranks.

        # Eq. (12)
        self.U = self.U @ U_[:r, :]
        # Eq. (5)
        self.S = S_


class RainflowCounting(Kirt_base):
    pass
