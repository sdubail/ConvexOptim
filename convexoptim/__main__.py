import numpy as np
from compute import barr_method


def optimise(
    X: np.ndarray, y: np.ndarray, lamda: float, v0: np.ndarray, eps: float, mu: float
) -> list[np.ndarray]:
    # Problem dimensions
    n, d = X.shape

    # Problem variables
    Q = (1 / 2) * np.eye(N=n)
    p = y
    A = np.vstack((X.T, -X.T))
    b = lamda * np.ones(2 * d)

    v_seq = barr_method(Q, p, A, b, v0, eps, mu)

    return v_seq


def main():
    X, y = gen_data(n, d)
    lamda = 10
    eps = 0.01
    v0 = np.zeros(X.shape[0])
    v_seq = optimise(X, y, lamda, v0, eps, mu)
    + NEED TO STOCK FUNCTION EVALUATIONS ALONG V_SEQ TO PLOT f(V_seq) - f(V_seq[-1])