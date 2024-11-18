import matplotlib.pyplot as plt
import numpy as np
from compute import barr_method


def plot_convergence(f_seq: list[float]) -> None:
    """
    Plot the convergence of the objective function values relative to the final value.

    Args:
        f_seq: List of objective function values from each iteration
    """
    # Convert to numpy array for easier manipulation
    f_array = np.array(f_seq)

    # Calculate difference from final value
    f_diff = f_array - f_array[-1]

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_diff, "b-", linewidth=2, marker="o", markersize=4)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Add labels and title
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("$f(x_k) - f(x^*)$", fontsize=12)
    plt.title("Convergence Plot", fontsize=14)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def optimise(
    X: np.ndarray, y: np.ndarray, lamda: float, v0: np.ndarray, eps: float, mu: float
) -> tuple[list[np.ndarray], list[float]]:
    # Problem dimensions
    n, d = X.shape

    # Problem variables
    Q = (1 / 2) * np.eye(N=n)
    p = y
    A = np.vstack((X.T, -X.T))
    b = lamda * np.ones(2 * d)

    v_seq, f_seq = barr_method(Q, p, A, b, v0, eps, mu)

    return v_seq, f_seq


def gen_data(n: int, d: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for the optimization problem.

    Args:
        n: Number of samples
        d: Number of features/dimensions
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where:
        - X is a n×d matrix of features
        - y is a n-dimensional vector of labels/targets
    """
    np.random.seed(seed)

    # Generate random features
    X = np.random.randn(n, d)

    # Generate true coefficients
    true_coeffs = np.random.randn(d)

    # Generate target variable with some noise
    noise = np.random.normal(0, 0.1, n)
    y = X @ true_coeffs + noise

    return X, y


def main() -> None:
    # For the main function, a typical value for mu would be:
    mu = 10.0  # Standard value used in barrier methods
    lamda = 10
    eps = 0.01
    # Example usage:
    n = 100  # number of samples
    d = 20  # number of features
    X, y = gen_data(n, d)

    v0 = np.zeros(X.shape[0])
    v_seq, f_seq = optimise(X, y, lamda, v0, eps, mu)
    plot_convergence(f_seq)


if __name__ == "__main__":
    main()
