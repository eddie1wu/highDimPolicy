import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_rct
from hdpolicy.gen_data import gen_logistic
from hdpolicy.linear_classifier import gd

def main():

    n, n_dim = 100, 120
    rct_p = 0.5

    rng = make_rng(890)

    beta_0 = rng.normal(-1, 2, size=(n_dim, 1))
    beta_1 = rng.normal(2, 2, size=(n_dim, 1))

    Y, X, D, Y0, Y1 = gen_rct(n, n_dim, rct_p, rng, beta_0=beta_0,
                              beta_1=beta_1)

    target_train = np.sign(Y) * D
    weight = np.abs(Y) / (D * rct_p + (1 - D) / 2)

    alpha_0 = rng.random((n_dim, 1))

    print("START")

    ### Run SVM directly
    svc = LinearSVC(
        C=1e10,                 # very large
        loss="hinge",
        fit_intercept=False,
        tol=1e-6,
        max_iter=50000
    )
    svc.fit(X, target_train.ravel())
    w_svc = svc.coef_.ravel()
    w_svc = w_svc / np.linalg.norm(w_svc)

    print("FINISHED SVC")


    ### With weight
    alpha_w_hat = gd(alpha_0, X, target_train, weight)
    alpha_w_hat = alpha_w_hat / np.linalg.norm(alpha_w_hat)

    print("FINISHED W")

    ### Without weight
    alpha_hat = gd(alpha_0, X, target_train)
    alpha_hat = alpha_hat / np.linalg.norm(alpha_hat)

    print("FINISHED NO W")



    ### Cosine distance
    print("SVC vs GD weighted")
    print(alpha_w_hat.T @ w_svc)

    print("SVC vs GD unweighted")
    print(alpha_hat.T @ w_svc)

    print("GD weighted vs GD unweighted")
    print(alpha_hat.T @ alpha_w_hat)

    print(alpha_w_hat)
    print(alpha_hat)
    print(w_svc)


    # # Generate data
    # n, n_dim = 100, 120
    # rng = make_rng(123)
    # beta = rng.normal(0, 1, size = n_dim).reshape(-1, 1)
    # X, y, _ = gen_logistic(n, n_dim, rng, beta)
    # print(X.shape)
    # print(y.shape)
    # print(beta.shape)
    #
    # # # Visualize in 2D
    # # X1_boundary = np.linspace(-4, 3, 100)
    # # X2_boundary = -beta[0] / beta[1] * X1_boundary
    # #
    # # plt.figure(figsize=(5, 5))
    # # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    # # plt.plot(X1_boundary, X2_boundary, color='green', linestyle='--')
    # # plt.show()
    #
    # # Estimate beta by GD
    # alpha_0 = rng.random((n_dim, 1))
    # weight = rng.uniform(0, 100, (n, 1))
    #
    # print("START")
    #
    # ### With weight
    # alpha_w_hat = gd(alpha_0, X, y, weight)
    # alpha_w_hat = alpha_w_hat / np.linalg.norm(alpha_w_hat)
    #
    # print("FINISHED W")
    #
    # ### Without weight
    # alpha_hat = gd(alpha_0, X, y)
    # alpha_hat = alpha_hat / np.linalg.norm(alpha_hat)
    #
    # print("FINISHED NO W")
    #
    # ### Cosine distance
    # print(alpha_hat.T @ alpha_w_hat)
    # print(alpha_w_hat)
    # print(alpha_hat)


if __name__ == '__main__':
    main()


# 0.95930241
# 0.96248132
# 0.96905123
# 0.95828208
# 0.95833164


