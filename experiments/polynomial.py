import os
import itertools

import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from numpy.polynomial.hermite_e import hermevander
import math
from sklearn.svm import LinearSVC

from tqdm import tqdm

import matplotlib.pyplot as plt

from hdpolicy.config import Config
from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_polynomial
from hdpolicy.linear_classifier import gd
from hdpolicy.metrics import compute_welfare



def run_polynomial(cfg):

    rng = make_rng(cfg.seed)

    # Make data
    Y, X, D, Y0, Y1, tau = gen_polynomial(
        cfg.n_train,
        cfg.rct_probability,
        rng
    )

    # # Create polynomial features
    # degree = 10
    # poly = PolynomialFeatures(degree=degree, include_bias=True)
    # X_phi = poly.fit_transform(X)
    #
    # # Standardize each feature
    # scaler = StandardScaler(with_mean=True, with_std=True)
    # xx_phi = X_phi.copy()
    # xx_phi[:, 1:] = scaler.fit_transform(xx_phi[:, 1:])  # keep bias column unchanged

    degree = 30
    x_phi = hermite_sieve_1d(X, degree)
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_phi[:, 1:] = scaler.fit_transform(x_phi[:, 1:])  # keep bias column unchanged


    # Estimate param
    target_train = np.sign(Y) * D
    weight = np.abs(Y) / (D * cfg.rct_probability + (1 - D) / 2)

    # print(x_phi)

    # Compute the SVC
    svc = LinearSVC(
        C = 1e10,  # very large
        loss = "hinge",
        fit_intercept = True,
        tol=1e-6,
        max_iter = 1000000
    )
    svc.fit(x_phi, target_train.ravel())
    w_svc = svc.coef_.ravel()

    print("FINISHED SVC")
    print(w_svc)


    # # Compute param
    # alpha_0 = rng.random((x_phi.shape[1], 1))
    # alpha_hat = gd(alpha_0, x_phi, target_train, weight, plotting = True)
    # alpha_hat = gd(alpha_0, x_phi, target_train, weight)
    # print(alpha_hat)


    # Compute classification error in sample
    # pred = np.sign(x_phi @ alpha_hat)
    # pred = np.sign(x_phi @ w_svc)
    pred = svc.predict(x_phi)
    acc = np.mean(target_train == pred)

    print(f"In sample accuracy is {acc: .5f}")


    # # X grid
    # x_grid = np.linspace(-2,2,200).reshape(-1,1)
    # # x_grid_phi = poly.fit_transform(x_grid)
    # x_grid_phi = hermite_sieve_1d(x_grid, degree)
    #
    # fitted_fun = x_grid_phi @ w_svc
    #
    # plt.plot(x_grid, fitted_fun, label = "fitted function")
    # plt.scatter(X, Y, label = 'Y')
    # plt.scatter(X, D, label = 'D')
    # plt.axhline(0, color = "black", alpha = 0.5, ls = "--")
    # plt.axvline(0, color="black", alpha=0.5, ls="--")
    # plt.ylim(-4, 9)
    # plt.legend()
    # plt.show()






    # Xv = np.vander(X.ravel(), N=20, increasing=True)  # columns: 1, x, ..., x^19
    #
    # beta = np.linalg.solve(Xv, target_train.astype(float))
    # f = Xv @ beta
    #
    # print("max interpolation error:", np.max(np.abs(f - target_train)))
    # print("train acc:", np.mean(np.sign(f) == target_train))
    # print("condition number:", np.linalg.cond(Xv))


def hermite_sieve_1d(x, degree):

    x = np.asarray(x).reshape(-1)

    Phi = hermevander(x, degree)

    # from scipy.special import gammaln
    #
    # k = np.arange(degree + 1)
    # scales = np.exp(0.5 * gammaln(k + 1))
    #
    # # scales = np.sqrt([math.factorial(k) for k in range(degree + 1)])
    #
    # Phi = Phi / scales

    return Phi
