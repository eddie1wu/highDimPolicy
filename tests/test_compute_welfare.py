import numpy as np
import matplotlib.pyplot as plt

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_rct
from hdpolicy.linear_classifier import gd

def main():

    # Generate data
    n, n_dim = 6, 2
    rng = make_rng(12)
    beta_0 = rng.normal(-1, 1, size = n_dim).reshape(-1, 1)
    beta_1 = rng.normal(1, 1, size=n_dim).reshape(-1, 1)
    y, X, D, y0, y1 = gen_rct(n, n_dim, 0.5, rng, beta_0, beta_1)
    D = D.ravel()

    print(D)
    print(y)
    print(list(zip(y0, y1)))

    beta = beta_1 - beta_0

    # Visualize in 2D
    X1_boundary = np.linspace(-2, 2, 100)
    X2_boundary = -beta[0] / beta[1] * X1_boundary

    plt.figure(figsize=(5, 5))
    plt.scatter(X[D == 1, 0], X[D == 1, 1], color='red', label='D = +1', alpha=0.7)
    plt.scatter(X[D == -1, 0], X[D == -1, 1], color='blue', label='D = -1', alpha=0.7)
    plt.plot(X1_boundary, X2_boundary, color='green', linestyle='--')
    plt.legend()
    plt.show()
    #
    # # Estimate beta by GD
    # beta_0 = rng.random((n_dim, 1))
    # beta_hat = gd(beta_0, X, y)
    #
    # print(f"True param: \n {beta}")
    # print(f"Fitted param: \n {beta_hat}")
    # print(f"Distance between true and fitted: \n {np.linalg.norm(beta - beta_hat, ord = 1) / n_dim}")
    #
    # print("Finished")


if __name__ == '__main__':
    main()

