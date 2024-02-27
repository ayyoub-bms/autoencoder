import numpy as np
from scipy import stats

T = 180
N = 200
Px = Pc = 50
K = 3

l_theta = np.array([1.2, 1, 0.8])
n_theta = np.array([1, 2, 0.6])


def empty(*args):
    return np.empty(args, dtype=np.float32)


def simulate(T=T, N=N, Px=Px, Pc=Pc, K=K):
    l_betas = empty(T, N, K)
    n_betas = empty(T, N, K)

    l_returns = empty(T, N)
    n_returns = empty(T, N)

    l_managed = empty(T, 1, Px)
    n_managed = empty(T, 1, Px)

    characteristics = empty(T, N, Pc)

    W = np.zeros((K, Px))
    for i in range(K):
        W[i, i] = 1

    mu = 3e-2 * np.ones(Px)
    cov = 1e-2 * np.eye(Px)
    x = np.random.multivariate_normal(mu, cov, size=T)

    mu = np.zeros(K)
    cov = 1e-4 * np.eye(K)
    eta = np.random.multivariate_normal(mu, cov, size=T)

    factors = x @ W.T + eta

    eps = stats.t(loc=0, scale=.1, df=5).rvs((T, N))
    rho = np.random.uniform(0.9, 1, Pc)

    c_bar = 0

    for t in range(T):
        c_bar = c_bar * rho + np.random.randn(N, Pc)
        ct = 2 * stats.rankdata(c_bar, axis=0) / (N + 1) - 1

        characteristics[t] = ct

        l_betas[t] = lb = ct[:, :3] * l_theta

        n_betas[t] = nb = np.array([
            ct[:, 0] * ct[:, 0],
            ct[:, 0] * ct[:, 1],
            np.sign(ct[:, 2])
        ]).T * n_theta

        l_returns[t] = lr = lb @ factors[t] + eps[t]
        n_returns[t] = nr = nb @ factors[t] + eps[t]

        ctc = ct.T @ ct
        l_managed[t] = np.linalg.solve(ctc, (ct.T @ lr).reshape(-1, 1)).T
        n_managed[t] = np.linalg.solve(ctc, (ct.T @ nr).reshape(-1, 1)).T
    return (
        n_betas,
        l_betas,
        n_returns,
        l_returns,
        n_managed,
        l_managed,
        characteristics,
        factors
    )
