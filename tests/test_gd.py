import numpy as np
import matplotlib.pyplot as plt

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_logistic
from hdpolicy.linear_classifier import gd


def main():

    # Generate data
    n, n_dim = 500, 20
    rng = make_rng(123)
    beta = rng.normal(0, 1, size = n_dim).reshape(-1, 1)
    X, y, _ = gen_logistic(n, n_dim, rng, beta)
    print(X.shape)
    print(y.shape)
    print(beta.shape)

    # Visualize in 2D
    X1_boundary = np.linspace(-4, 3, 100)
    X2_boundary = -beta[0] / beta[1] * X1_boundary

    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.plot(X1_boundary, X2_boundary, color='green', linestyle='--')
    plt.show()

    # Estimate beta by GD
    beta_0 = rng.random((n_dim, 1))
    beta_hat = gd(beta_0, X, y)

    print(f"True param: \n {beta}")
    print(f"Fitted param: \n {beta_hat}")
    print(f"Distance between true and fitted: \n {np.abs(beta - beta_hat)}")

    print("Finished")


if __name__ == '__main__':
    main()
