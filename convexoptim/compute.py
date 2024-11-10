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
    # Compute Av for constraints
    Av = A @ v

    # Initialize gradient components
    grad = t * (2 * Q @ v + p)  # Term from quadratic objective

    # Add logarithmic barrier terms
    for i in range(len(b)):
        ai = A[i, :]  # i-th row of A
        grad += ai / (b[i] - Av[i])

    # Compute Hessian inverse
    hess = t * 2 * Q  # Term from quadratic objective

    # Add barrier terms to Hessian
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
) -> float:
    """
    Perform backtracking line search to find step size.

    Args:
        f_eval: Current function value
        grad: Current gradient
        p: Search direction
        alpha: Line search parameter (default: 0.01)
        beta: Line search parameter (default: 0.8)

    Returns:
        Step size t
    """
    t = 1.0
    while f_eval(t) > f_eval(0) + alpha * t * grad.T @ p:
        t *= beta
    return t


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
) -> list[np.ndarray]:
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

    while True:
        # 1. Compute Newton step and decrement
        grad, hess_inv = compute_gradient_hessian(Q, p, A, b, t, v)
        delta_v = -hess_inv @ grad
        lambda_sq = -grad.T @ delta_v

        # 2. Stopping criterion
        if lambda_sq / 2 <= eps:
            break

        # 3. Line search
        def f_eval(step: float, v: np.ndarray, delta_v: np.ndarray) -> float:
            return evaluate_objective(Q, p, A, b, t, v + step * delta_v)

        step_size = backtracking_line_search(f_eval, grad, delta_v)

        # 4. Update
        v = v + step_size * delta_v
        v_seq.append(v.copy())

    return v_seq


def barr_method(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    v0: np.ndarray,
    eps: float,
    mu: float,
) -> list[np.ndarray]:
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
    # Initialize parameters
    t = 1.0
    v = v0.copy()
    v_seq = [v.copy()]
    m = len(b)  # Number of inequality constraints

    while m / t > eps:
        # Solve centering step
        v_centering = centering_step(Q, p, A, b, t, v, eps)
        v = v_centering[-1]
        v_seq.extend(v_centering)

        # Update t
        t *= mu

    return v_seq
