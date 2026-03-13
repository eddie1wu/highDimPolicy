import numpy as np

import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from hdpolicy.config import Config
from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_rct
from hdpolicy.utils import split_train_test, make_weight_and_target
from hdpolicy.linear_classifier import gd
from hdpolicy.metrics import compute_welfare


def run_simple_linear(cfg: Config, max_workers: int | None = None):

    max_workers = max_workers if max_workers is not None else os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(executor.map(run_rep, itertools.repeat(cfg), range(cfg.n_rep)))

    # Unpack results
    (train_welfare, test_welfare, oracle_welfare, p_grid) = zip(*results_list)

    # Average over reps
    train_welfare = np.mean(train_welfare, axis=0)
    test_welfare = np.mean(test_welfare, axis=0)
    oracle_welfare = np.mean(oracle_welfare, axis = 0)

    p_grid = p_grid[0]

    return {
        'train_welfare': train_welfare,
        'test_welfare': test_welfare,
        'oracle_welfare': oracle_welfare,
        'p_grid': p_grid
    }



def run_rep(cfg: Config, rep: int = 0):

    rng = make_rng(cfg.seed + rep)

    # Create feature grid
    p_grid = np.arange(5, cfg.dim_max + 1, 10)

    # Store results
    train_welfare = np.zeros(len(p_grid))
    test_welfare = np.zeros(len(p_grid))
    oracle_welfare = np.zeros(len(p_grid))

    for idx in tqdm( range(len(p_grid)) ):

        # Gen current p_grid value data
        p = p_grid[idx]
        # Generate true params
        beta_0 = rng.normal(-1, 2, size=(p, 1))
        beta_1 = rng.normal(1, 2, size=(p, 1))


        beta_0 /= np.linalg.norm(beta_0)
        beta_1 /= np.linalg.norm(beta_1)

        Y, X, D, Y0, Y1 = gen_rct(
            cfg.n_train + cfg.n_test,
            p,
            cfg.rct_probability,
            rng,
            beta_0=beta_0,
            beta_1=beta_1
        )

        # Train test split
        X_train, Y_train, D_train, Y0_train, Y1_train, \
            X_test, _, _, Y0_test, Y1_test = split_train_test(cfg.n_train, X, Y, D, Y0, Y1)

        # Create the classification weights and target
        weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)

        # Compute param
        alpha_0 = rng.random((p, 1))
        alpha_hat = gd(alpha_0, X_train, target_train, weight)

        # Compute welfare
        train_welfare[idx] += compute_welfare(X_train @ alpha_hat, Y0_train, Y1_train)
        test_welfare[idx] += compute_welfare(X_test @ alpha_hat, Y0_test, Y1_test)
        oracle_welfare[idx] += compute_welfare(X_test @ (beta_1 - beta_0), Y0_test, Y1_test)

    print(f"Finished iteration {rep}.")

    return (train_welfare, test_welfare, oracle_welfare, p_grid)



#
#
#
# def run_rep(cfg: Config, rep: int = 0):
#
#     rng = make_rng(cfg.seed + rep)
#
#     # Generate true params
#     beta_0 = rng.normal(-1, 2, size=(cfg.dim_max, 1))
#     beta_1 = rng.normal(1, 2, size=(cfg.dim_max, 1))
#
#     # Create feature grid
#     p_grid = np.arange(5, cfg.dim_max + 1, 10)
#
#     # Store results
#     train_welfare = np.zeros(len(p_grid))
#     test_welfare = np.zeros(len(p_grid))
#     oracle_welfare = np.zeros(len(p_grid))
#
#     for idx in tqdm( range(len(p_grid)) ):
#
#         # Gen current p_grid value data
#         p = p_grid[idx]
#         beta_0_curr = beta_0[:p]
#         beta_1_curr = beta_1[:p]
#
#         beta_0_curr /= np.linalg.norm(beta_0_curr)
#         beta_1_curr /= np.linalg.norm(beta_1_curr)
#
#         Y, X, D, Y0, Y1 = gen_rct(
#             cfg.n_train + cfg.n_test,
#             p,
#             cfg.rct_probability,
#             rng,
#             beta_0=beta_0_curr,
#             beta_1=beta_1_curr
#         )
#
#         # Train test split
#         X_train, Y_train, D_train, Y0_train, Y1_train, \
#             X_test, _, _, Y0_test, Y1_test = split_train_test(cfg.n_train, X, Y, D, Y0, Y1)
#
#         # Create the classification weights and target
#         weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)
#
#         # Compute param
#         alpha_0 = rng.random((p, 1))
#         alpha_hat = gd(alpha_0, X_train, target_train, weight)
#
#         # Compute welfare
#         train_welfare[idx] += compute_welfare(X_train @ alpha_hat, Y0_train, Y1_train)
#         test_welfare[idx] += compute_welfare(X_test @ alpha_hat, Y0_test, Y1_test)
#         oracle_welfare[idx] += compute_welfare(X_test @ (beta_1_curr - beta_0_curr), Y0_test, Y1_test)
#
#     print(f"Finished iteration {rep}.")
#
#     return (train_welfare, test_welfare, oracle_welfare, p_grid)
#
#
