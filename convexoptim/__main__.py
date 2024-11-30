import matplotlib.pyplot as plt
import numpy as np
from compute import barr_method


def plot_convergence(
    f_seq_multi: list[list[float]],
    crit_seq_multi: list[list[float]],
    mu: list[float],
    n: float,
    d: float,
    lamda: float,
    eps: float,
    base_lr: float,
) -> None:
    """
    Plot both the criterion evolution and convergence to optimal value.

    Args:
        f_seq_multi: List of lists containing objective function values for computing difference from optimal
        crit_seq_multi: List of lists containing criterion values evolution
        mu: List of mu values used
        n, d, lamda, eps, base_lr: Parameters for filename
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First subplot: Criterion evolution
    for crit_seq, mu_i in zip(crit_seq_multi, mu, strict=False):
        crit_array = np.array(crit_seq)
        ax1.semilogy(
            crit_array,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"μ={mu_i}",
        )

    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xlabel("Centering iterations", fontsize=12)
    ax1.set_ylabel("Criterion value", fontsize=12)
    ax1.set_title("Evolution of precision criterion", fontsize=14)
    ax1.legend()

    # Second subplot: Difference from optimal value
    for f_seq, mu_i in zip(f_seq_multi, mu, strict=False):
        f_array = np.array(f_seq)
        f_diff = f_array - f_array[-1]
        ax2.semilogy(
            f_diff,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"μ={mu_i}",
        )

    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlabel("Total Newton iterations", fontsize=12)
    ax2.set_ylabel("$f(x_k) - f(x^*)$", fontsize=12)
    ax2.set_title("Convergence to optimal value", fontsize=14)
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save figure
    plt.savefig(
        f"convergence_n_{n}_d_{d}_lamda_{lamda}_eps_{eps}_baselr_{base_lr}.png",
        bbox_inches="tight",
        dpi=300,
    )

    # Close the figure to free memory
    plt.close()


def optimise(
    X: np.ndarray,
    y: np.ndarray,
    lamda: float,
    v0: np.ndarray,
    eps: float,
    mu: float,
    base_lr: float,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    # Problem dimensions
    n, d = X.shape

    # Problem variables
    Q = (1 / 2) * np.eye(N=n)
    p = y
    A = np.vstack((X.T, -X.T))
    b = lamda * np.ones(2 * d)

    v_seq, f_seq, crit_seq = barr_method(Q, p, A, b, v0, eps, mu, base_lr)

    return v_seq, f_seq, crit_seq


def gen_data(
    n: int, d: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        - true_coeff is a d-dimensional vector of coefficients
    """
    np.random.seed(seed)

    # Generate random features
    X = np.random.randn(n, d)

    # Generate true coefficients
    true_coeffs = np.random.randn(d)

    # Generate target variable with some noise
    noise = np.random.normal(0, 0.1, n)
    y = X @ true_coeffs + noise

    return X, y, true_coeffs


def main() -> None:
    mu = [2.0, 5.0, 15.0, 50.0, 100.0, 200.0, 500.0]
    lamda = 10
    eps = 0.01
    base_lr = 1
    n = 100  # number of samples
    d = 20  # number of features
    X, y, true_w = gen_data(n, d)

    v0 = np.zeros(X.shape[0])  # not sure that 0 is a good idea
    f_seq_multi = []
    v_seq_multi = []
    crit_seq_multi = []
    for mu_i in mu:
        print(mu_i)
        simul = optimise(X, y, lamda, v0, eps, mu_i, base_lr)
        f_seq_multi.append(simul[1])
        v_seq_multi.append(simul[0])
        crit_seq_multi.append(simul[2])

    plot_convergence(f_seq_multi, crit_seq_multi, mu, n, d, lamda, eps, base_lr)

    # At optimum :
    # v = z = Xw - y
    # X.T @ (v + y) = X.T @ Xw
    # (X.T @ X)^-1 @ X.T (v + y) = w
    plt.figure(figsize=(10, 6))

    # Calculate errors
    w_err = []
    for v_seq in v_seq_multi:
        v_opti = v_seq[-1]
        w = np.linalg.pinv(X.T @ X) @ X.T @ (v_opti + y)
        w_err.append(np.sum(((w - true_w) ** 2) / true_w**2) / d)

    # Create enhanced plot
    plt.plot(
        mu,
        w_err,
        "o-",
        linewidth=2,
        markersize=6,
        color="#2E86C1",
        label="Weight Error",
    )

    # Add labels and title
    plt.xlabel("μ Parameter", fontsize=12)
    plt.ylabel("Normalized Weight Error", fontsize=12)
    plt.title("Weight Error vs. μ Parameter", fontsize=14, pad=15)

    # Customize grid and layout
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)

    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.savefig("w_err_mu.png")


if __name__ == "__main__":
    main()
