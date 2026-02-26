import os
import itertools

import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

from hdpolicy.config import Config
from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_nonlinear
from hdpolicy.linear_classifier import gd
from hdpolicy.metrics import compute_welfare

def run_misspecified(cfg: Config, max_workers: int | None = None):

    max_workers = max_workers if max_workers is not None else os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(executor.map(run_rep, itertools.repeat(cfg), range(cfg.n_rep)))

    # Unpack results
    # (train_welfare, test_welfare, oracle_welfare, true_feature_welfare, p_grid) = zip(*results_list)
    (train_welfare, test_welfare, oracle_welfare, p_grid) = zip(*results_list)

    train_welfare = np.array(train_welfare)
    test_welfare = np.array(test_welfare)
    p_grid = p_grid[0]
    oracle_welfare = np.mean(oracle_welfare)
    # true_feature_welfare = np.mean(true_feature_welfare)

    train_welfare = np.mean(train_welfare, axis = 0)
    test_welfare = np.mean(test_welfare, axis = 0)

    return {
        'train_welfare': train_welfare,
        'test_welfare': test_welfare,
        'oracle_welfare': oracle_welfare,
        # 'true_feature_welfare': true_feature_welfare,
        'p_grid': p_grid
    }


def run_rep(cfg: Config, rep: int = 0):

    # Generate data
    rng = make_rng(cfg.seed + rep)

    Y, X, D, Y0, Y1, tau, true_form = gen_nonlinear(cfg.n_train+cfg.n_test, cfg.rct_probability, rng, dgp = 'friedman')

    # Build polynomial features
    poly = PolynomialFeatures(degree = cfg.degree, include_bias = True)
    phi = poly.fit_transform(X)

    # Standardize the features except the constant
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_phi = phi.copy()
    X_phi[:, 1:] = scaler.fit_transform(X_phi[:, 1:])  # keep bias column unchanged

    # Define grid of feature dimensions
    p_grid = np.arange(5, X_phi.shape[1]+1, 10)

    # Store results
    train_welfare = np.full(len(p_grid), 0.0)
    test_welfare = np.full(len(p_grid), 0.0)
    oracle_welfare = 0

    # Loop over p_grid
    for idx in tqdm(range(len(p_grid))):
    # for idx in range(len(p_grid)):

        p = p_grid[idx]
        temp = X_phi[:,:p]

        X_train, X_test = temp[:cfg.n_train], temp[cfg.n_train:]
        Y_train, D_train = Y[:cfg.n_train], D[:cfg.n_train]

        Y0_train, Y0_test = Y0[:cfg.n_train], Y0[cfg.n_train:]
        Y1_train, Y1_test = Y1[:cfg.n_train], Y1[cfg.n_train:]

        target_train = np.sign(Y_train) * D_train
        weight = np.abs(Y_train) / (D_train * cfg.rct_probability + (1 - D_train) / 2)

        # Compute param
        alpha_0 = rng.random((p, 1))
        alpha_hat = gd(alpha_0, X_train, target_train, weight)

        # Compute welfare
        train_welfare[idx] += compute_welfare(X_train @ alpha_hat, Y0_train, Y1_train)
        test_welfare[idx] += compute_welfare(X_test @ alpha_hat, Y0_test, Y1_test)

    oracle_welfare += compute_welfare(tau[cfg.n_train:], Y0_test, Y1_test)

    #
    # ### Use the true DGP, but limited sample
    # n_main, n_pair, n_triple, n_quad, max_power = 6, 6, 4, 3, 3
    # (all_pairs, powers, all_triplets, all_quads) = true_form
    # X_true = []
    #
    # for j in range(n_main):  # main effect
    #     for p in np.arange(max_power):
    #         X_true.append( X[:, j].reshape(-1, 1) ** p )
    #
    # for ((i, j), (p, q)) in zip(all_pairs, powers):
    #     X_true.append( (X[:, i] ** p).reshape(-1, 1) * (X[:, j] ** q).reshape(-1, 1) )
    #
    # for (i, j, k) in all_triplets:
    #     X_true.append( X[:, i].reshape(-1, 1) * X[:, j].reshape(-1, 1) * X[:, k].reshape(-1, 1) )
    #
    # for (i,j,k,l) in all_quads:
    #     X_true.append(
    #         X[:, i].reshape(-1, 1)
    #         * X[:, j].reshape(-1, 1)
    #         * X[:, k].reshape(-1, 1)
    #         * X[:, l].reshape(-1, 1)
    #     )
    #
    # X_true = np.hstack(X_true)
    # X_train, X_test = X_true[:cfg.n_train], X_true[cfg.n_train:]
    #
    # # Compute param
    # alpha_0 = rng.random((X_true.shape[1], 1))
    # alpha_hat = gd(alpha_0, X_train, target_train, weight)
    #
    # # Compute welfare
    # true_feature_welfare = compute_welfare(X_test @ alpha_hat, Y0_test, Y1_test)

    # return (train_welfare, test_welfare, oracle_welfare, true_feature_welfare, p_grid)
    return (train_welfare, test_welfare, oracle_welfare, p_grid)



# def run_misspecified(cfg: Config, rep: int = 0):
#
#     # Generate data
#     rng = make_rng(cfg.seed + rep)
#
#     # # Store results
#     # train_welfare_all = np.full(len(p_grid), np.nan)
#     # test_welfare_all = np.full(len(p_grid), np.nan)
#     oracle_welfare = 0
#
#     for t in tqdm(range(cfg.n_rep)):
#
#         Y, X, D, Y0, Y1, tau = gen_nonlinear(cfg.n_train+cfg.n_test, cfg.rct_probability, rng, dgp = 'poly_interactions')
#
#         # Build polynomial features
#         poly = PolynomialFeatures(degree = cfg.degree, include_bias = True)
#         phi = poly.fit_transform(X)
#
#         # Standardize the features except the constant
#         scaler = StandardScaler(with_mean=True, with_std=True)
#         X_phi = phi.copy()
#         X_phi[:, 1:] = scaler.fit_transform(X_phi[:, 1:])  # keep bias column unchanged
#
#         # Define grid of feature dimensions
#         p_grid = np.arange(5, X_phi.shape[1]+1, 10)
#
#         if t == 0:
#             # Store results
#             train_welfare = np.full(len(p_grid), 0.0)
#             test_welfare = np.full(len(p_grid), 0.0)
#
#         # Loop over p_grid
#         # for idx in tqdm(range(len(p_grid))):
#         for idx in range(len(p_grid)):
#
#             p = p_grid[idx]
#             temp = X_phi[:,:p]
#
#             X_train, X_test = temp[:cfg.n_train], temp[cfg.n_train:]
#             Y_train, D_train = Y[:cfg.n_train], D[:cfg.n_train]
#
#             Y0_train, Y0_test = Y0[:cfg.n_train], Y0[cfg.n_train:]
#             Y1_train, Y1_test = Y1[:cfg.n_train], Y1[cfg.n_train:]
#
#             target_train = np.sign(Y_train) * D_train
#             weight = np.abs(Y_train) / (D_train * cfg.rct_probability + (1 - D_train) / 2)
#
#             # Compute param
#             alpha_0 = rng.random((p, 1))
#             alpha_hat = gd(alpha_0, X_train, target_train, weight)
#
#             # Compute welfare
#             train_welfare[idx] += compute_welfare(X_train @ alpha_hat, Y0_train, Y1_train)
#             test_welfare[idx] += compute_welfare(X_test @ alpha_hat, Y0_test, Y1_test)
#
#         oracle_welfare += compute_welfare(tau[cfg.n_train:], Y0_test, Y1_test)
#
#
#     train_welfare /= cfg.n_rep
#     test_welfare /= cfg.n_rep
#     oracle_welfare /= cfg.n_rep
#
#     return (train_welfare, test_welfare, oracle_welfare, p_grid)
