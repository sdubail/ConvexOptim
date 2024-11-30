from collections.abc import Callable

import numpy as np


def compute_gradient_hessian(
    Q: np.ndarray, p: np.ndarray, A: np.ndarray, b: np.ndarray, t: float, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient and Hessian inverse of the barrier function.

    Args:
        Q: Positive semidefinite matrix
        p: Linear term vector
        A: Constraint matrix
        b: Constraint vector
        t: Barrier parameter
        v: Current point

    Returns:
        Tuple of (gradient, inverse_hessian)
    """
    Av = A @ v
    grad = t * (2 * Q @ v + p)

    for i in range(len(b)):
        ai = A[i, :]  # i-th row of A
        grad += ai / (b[i] - Av[i])

    hess = t * 2 * Q

    for i in range(len(b)):
        ai = A[i, :]
        hess += np.outer(ai, ai) / (b[i] - Av[i]) ** 2

    return grad, np.linalg.inv(hess)


def backtracking_line_search(
    f_eval: Callable,
    grad: np.ndarray,
    p: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.8,
    base_lr: float = 1.0,
) -> tuple[float, float]:
    """
    Perform backtracking line search to find step size.

    Args:
        f_eval: Current function value
        grad: Current gradient
        p: Search direction
        alpha: Line search parameter (default: 0.01)
        beta: Line search parameter (default: 0.8)
        base_lr: Line search parameter (default: 1.0)

    Returns:
        Step size t, function value at given step
    """
    t_step = base_lr  # really sensitive : can go in places where log is undefined
    f_0, f_t = f_eval(0), f_eval(t_step)
    while f_t > f_0 + alpha * t_step * grad.T @ p or np.isnan(f_t):
        t_step *= beta
        f_t = f_eval(t_step)
    return t_step, f_t


def evaluate_objective(
    Q: np.ndarray, p: np.ndarray, A: np.ndarray, b: np.ndarray, t: float, v: np.ndarray
) -> float:
    """
    Evaluate the barrier objective function.
    """
    quad_term = t * (v.T @ Q @ v + p.T @ v)
    barrier_term = -np.sum(np.log(b - A @ v))
    return quad_term + barrier_term


def centering_step(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    t: float,
    v0: np.ndarray,
    eps: float,
    base_lr: float,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Implement Newton method for the centering step.

    Args:
        Q: Positive semidefinite matrix
        p: Linear term vector
        A: Constraint matrix
        b: Constraint vector
        t: Barrier parameter
        v0: Initial point
        eps: Target precision

    Returns:
        List of iterates
    """
    v = v0.copy()
    v_seq = [v.copy()]
    f_seq: list[float] = []

    while True:
        grad, hess_inv = compute_gradient_hessian(Q, p, A, b, t, v)
        delta_v = -hess_inv @ grad
        lambda_sq = -grad.T @ delta_v

        if lambda_sq / 2 <= eps:
            break

        def f_eval(
            step: float, v: np.ndarray = v, delta_v: np.ndarray = delta_v
        ) -> float:
            return evaluate_objective(Q, p, A, b, t, v + step * delta_v)

        step_size, f_value = backtracking_line_search(
            f_eval, grad, delta_v, base_lr=base_lr
        )
        v = v + step_size * delta_v
        v_seq.append(v.copy())
        f_seq.append(f_value)

    return v_seq, f_seq


def barr_method(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    v0: np.ndarray,
    eps: float,
    mu: float,
    base_lr: float,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    Implement the barrier method for QP.

    Args:
        Q: Positive semidefinite matrix
        p: Linear term vector
        A: Constraint matrix
        b: Constraint vector
        v0: Initial feasible point
        eps: Target precision

    Returns:
        List of iterates
    """
    t = 1.0
    v = v0.copy()
    v_seq = [v.copy()]
    f_seq: list[float] = []
    m = len(b)
    crit_seq: list[float] = [m / t]
    while m / t > eps:
        v_centering, f_centering = centering_step(Q, p, A, b, t, v, eps, base_lr)
        v = v_centering[-1]

        v_seq.extend(v_centering)
        f_seq.extend(f_centering)

        t *= mu
        crit_seq.append(m / t)

    return v_seq, f_seq, crit_seq
