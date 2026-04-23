from pathlib import Path

import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from sklearn.preprocessing import SplineTransformer

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_polynomial
from hdpolicy.utils import *
from hdpolicy.linear_classifier import gd, _logistic_loss
from hdpolicy.metrics import compute_welfare

from hdpolicy.io.save_load import load_config, save_config, make_run_dir, save_results



def main():

    cfg = load_config("configs/misspec_spline.toml")

    results = run_spline(cfg)

    run_dir = make_run_dir(Path("results"), tag="misspec_spline")
    save_results(run_dir, results)
    save_config(run_dir, cfg)

    print(f"Saved results to {run_dir}.")



def run_spline(cfg, max_workers: int | None = None):

    max_workers = max_workers if max_workers is not None else os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(executor.map( run_single_trial, itertools.repeat(cfg), range(cfg.n_rep) ))

    # Unpack results
    results_list = zip(*results_list)

    # Average over reps
    results_list = [np.nanmean(r, axis = 0) for r in results_list]

    train_welfare, test_welfare, misspec_X_welfare, correct_X_welfare, oracle_train_welfare, oracle_test_welfare, \
        train_loss, train_risk, test_risk, beta_norm, \
        svc_test_welfare, svc_w_norm, \
        p_grid = results_list

    return {
        'train_welfare': train_welfare,
        'test_welfare': test_welfare,
        'misspec_X_welfare': misspec_X_welfare,
        'correct_X_welfare': correct_X_welfare,
        'oracle_train_welfare': oracle_train_welfare,
        'oracle_test_welfare': oracle_test_welfare,
        'train_loss': train_loss,
        'train_risk': train_risk,
        'test_risk': test_risk,
        'beta_norm': beta_norm,
        'svc_test_welfare': svc_test_welfare,
        'svc_w_norm': svc_w_norm,
        'p_grid': p_grid
    }



def run_single_trial(cfg, trial_idx: int = 0):

    # Generate data
    rng = make_rng(cfg.seed + trial_idx)

    Y, X, D, Y0, Y1, tau = gen_polynomial(
        cfg.n_train + cfg.n_test,
        cfg.X_dim_max,
        cfg.rct_probability,
        rng
    )

    # Traint test split
    Y_train, X_train, D_train, Y0_train, Y1_train, tau_train, \
        Y_test, X_test, D_test, Y0_test, Y1_test, tau_test = split_train_test(cfg.n_train, Y, X, D, Y0, Y1, tau)

    # Create classification weight and target
    weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)
    _, target_test = make_weight_and_target(Y_test, D_test, cfg.rct_probability)

    # Define knot grid
    knot_grid = np.concatenate([
        np.arange(4, 89, 4),  # 4, 8, ..., 88
        np.arange(120, 401, 40)  # 120, 160, ..., 400
    ])

    p_grid = np.zeros(len(knot_grid))

    train_welfare = np.zeros(len(p_grid))
    test_welfare = np.zeros(len(p_grid))
    correct_X_welfare = 0
    misspec_X_welfare = 0
    oracle_train_welfare = 0
    oracle_test_welfare = 0
    train_loss = np.zeros(len(p_grid))
    train_risk = np.zeros(len(p_grid))
    test_risk = np.zeros(len(p_grid))
    beta_norm = np.zeros(len(p_grid))

    svc_test_welfare = np.zeros(len(p_grid))
    svc_w_norm = np.zeros(len(p_grid))

    for idx in tqdm(range(len(knot_grid))):

        n_knots = knot_grid[idx]
        spline = SplineTransformer(
            n_knots=n_knots,
            degree= cfg.spline_degree,
            include_bias=False
        )

        Phi_train_curr = spline.fit_transform(X_train)
        Phi_test_curr = spline.transform(X_test)

        p_grid[idx] = Phi_train_curr.shape[1]

        # Standardize and add intercept
        Phi_train_curr, Phi_test_curr = standardize_train_test(Phi_train_curr, Phi_test_curr)
        Phi_train_curr = add_intercept(Phi_train_curr)
        Phi_test_curr = add_intercept(Phi_test_curr)

        # Estimate param
        alpha_0 = rng.random((Phi_train_curr.shape[1], 1))
        alpha_hat = gd(alpha_0, Phi_train_curr, target_train, weight, lr=cfg.lr, steps=cfg.steps)

        # Compute welfare
        train_welfare[idx] = compute_welfare(Phi_train_curr @ alpha_hat, Y0_train, Y1_train)
        test_welfare[idx] = compute_welfare(Phi_test_curr @ alpha_hat, Y0_test, Y1_test)

        # Compute training loss
        train_loss[idx] = _logistic_loss(Phi_train_curr, target_train, alpha_hat, weight)

        # Compute classification risk
        train_risk[idx] = np.mean(np.sign(Phi_train_curr @ alpha_hat) != target_train)
        test_risk[idx] = np.mean(np.sign(Phi_test_curr @ alpha_hat) != target_test)

        # Compute norm of coefficient
        beta_norm[idx] = np.linalg.norm(alpha_hat)

        ### Run SVM
        svc_w_norm[idx] = np.nan
        svc_test_welfare[idx] = np.nan
        # if check_interpolation(Phi_train_curr, target_train):
        #     svc = LinearSVC(
        #         C=1e10,  # very large
        #         loss="hinge",
        #         fit_intercept = False,
        #         tol=1e-6,
        #         max_iter=1000000
        #     )
        #     svc.fit(Phi_train_curr, target_train.ravel())
        #     w_svc = svc.coef_.ravel()
        #     svc_w_norm[idx] = np.linalg.norm(w_svc)
        #
        #     # SVC welfare
        #     svc_test_welfare[idx] = compute_welfare(Phi_test_curr @ w_svc.reshape(-1, 1), Y0_test, Y1_test)

    # Misspecified full feature welfare
    X_train_curr = add_intercept(X_train)
    X_test_curr = add_intercept(X_test)
    alpha_0 = rng.random((X_train_curr.shape[1], 1))
    alpha_hat = gd(alpha_0, X_train_curr, target_train, weight, lr=cfg.lr, steps=cfg.steps)
    misspec_X_welfare += compute_welfare(X_test_curr @ alpha_hat, Y0_test, Y1_test)

    # Correctly specified full feature welfare
    X_train_curr = np.concatenate((X_train, X_train**2), axis = 1)
    X_test_curr = np.concatenate((X_test, X_test**2), axis = 1)
    X_train_curr = add_intercept(X_train_curr)
    X_test_curr = add_intercept(X_test_curr)
    alpha_0 = rng.random((X_train_curr.shape[1], 1))
    alpha_hat = gd(alpha_0, X_train_curr, target_train, weight, lr=cfg.lr, steps=cfg.steps)
    correct_X_welfare += compute_welfare(X_test_curr @ alpha_hat, Y0_test, Y1_test)

    # Oracle welfare
    oracle_train_welfare += compute_welfare(tau_train, Y0_train, Y1_train)
    oracle_test_welfare += compute_welfare(tau_test, Y0_test, Y1_test)

    print(f"Finished iteration {trial_idx}.")

    return (
        train_welfare, test_welfare, misspec_X_welfare, correct_X_welfare, oracle_train_welfare, oracle_test_welfare,
        train_loss, train_risk, test_risk, beta_norm,
        svc_test_welfare, svc_w_norm,
        p_grid
    )



if __name__ == "__main__":
    main()


