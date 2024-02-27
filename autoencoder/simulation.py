import numpy as np
from scipy import stats

T = 180
N = 200
Px = Pc = 50
K = 3


def empty(*args):
    return np.empty(args, dtype=np.float32)


def simulate():
    n_betas = empty(T, N, K)
    l_betas = empty(T, N, K)

    l_returns = empty(T, N)
    n_returns = empty(T, N)

    l_managed = empty(T, 1, Px)
    n_managed = empty(T, 1, Px)

    characteristics = empty(T, N, Pc)
    factors = empty(T, K)

    W = np.zeros((3, Px))
    for i in range(3):
        W[i, i] = 1

    rho = np.random.uniform(0.9, 1, (1, Pc))
    c_bar = 0

    for t in range(T):
        mu = 0.03 * np.ones(Px)
        cov = .01 * np.eye(Px)
        xt = np.random.multivariate_normal(mu, cov)

        mu = np.zeros(K)
        cov = .0001 * np.eye(K)
        etat = np.random.multivariate_normal(mu, cov)

        factors[t] = f = W @ xt + etat

        epst = np.random.randn(N, Pc)
        c_bar = c_bar * rho + epst
        ct = 2 * stats.rankdata(c_bar, axis=0) / (N + 1) - 1

        characteristics[t] = ct

        l_betas[t] = lb = np.array([
            1.2 * ct[:, 0],
            ct[:, 1],
            0.8 * ct[:, 2]
        ]).T

        n_betas[t] = nb = np.array([
            ct[:, 0] * ct[:, 0],
            2 * ct[:, 0] * ct[:, 1],
            .6 * np.sign(ct[:, 2])
        ]).T

        eps = stats.t(0, .1).rvs(5, size=N)
        n_returns[t] = nr = nb @ f + eps
        l_returns[t] = lr = lb @ f + eps

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
