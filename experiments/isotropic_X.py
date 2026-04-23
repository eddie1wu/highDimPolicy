from pathlib import Path

import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from sklearn.svm import LinearSVC

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_constant_signal
from hdpolicy.utils import split_train_test, make_weight_and_target, check_interpolation
from hdpolicy.linear_classifier import gd, _logistic_loss
from hdpolicy.linear_shrinkage import ShrinkageLogistic
from hdpolicy.metrics import compute_welfare

from hdpolicy.io.save_load import load_config, save_config, make_run_dir, save_results

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



def main():

    cfg = load_config("configs/isotropic_X.toml")

    results = run_isotropic(cfg)

    run_dir = make_run_dir(Path("results"), tag="isotropic_X")
    save_results(run_dir, results)
    save_config(run_dir, cfg)

    print(f"Saved results to {run_dir}.")



def run_isotropic(cfg, max_workers: int | None = None):

    max_workers = max_workers if max_workers is not None else os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(executor.map( run_single_trial, itertools.repeat(cfg), range(cfg.n_rep) ))

    # Unpack results
    results_list = zip(*results_list)

    # Average over reps
    results_list = [np.nanmean(r, axis = 0) for r in results_list]

    train_welfare, test_welfare, oracle_train_welfare, oracle_test_welfare, \
        train_loss, train_risk, test_risk, beta_norm, \
        svc_test_welfare, svc_w_norm, ridge_test_welfare, lasso_test_welfare, \
        p_grid = results_list

    return {
        'train_welfare': train_welfare,
        'test_welfare': test_welfare,
        'oracle_train_welfare': oracle_train_welfare,
        'oracle_test_welfare': oracle_test_welfare,
        'train_loss': train_loss,
        'train_risk': train_risk,
        'test_risk': test_risk,
        'beta_norm': beta_norm,
        'svc_test_welfare': svc_test_welfare,
        'svc_w_norm': svc_w_norm,
        'ridge_test_welfare': ridge_test_welfare,
        'lasso_test_welfare': lasso_test_welfare,
        'p_grid': p_grid
    }



def run_single_trial(cfg, trial_idx: int = 0):

    # Generate data
    rng = make_rng(cfg.seed + trial_idx)

    # Create feature grid
    p_grid = np.arange(5, cfg.dim_max + 1, 10)

    # Store outputs
    train_welfare = np.zeros(len(p_grid))
    test_welfare = np.zeros(len(p_grid))
    oracle_train_welfare = np.zeros(len(p_grid))
    oracle_test_welfare = np.zeros(len(p_grid))
    train_loss = np.zeros(len(p_grid))
    train_risk = np.zeros(len(p_grid))
    test_risk = np.zeros(len(p_grid))
    beta_norm = np.zeros(len(p_grid))

    svc_test_welfare = np.zeros(len(p_grid))
    svc_w_norm = np.zeros(len(p_grid))

    ridge_test_welfare = np.zeros(len(p_grid))
    lasso_test_welfare = np.zeros(len(p_grid))


    for idx in tqdm(range(len(p_grid))):

        p = p_grid[idx]

        Y, X, D, Y0, Y1, tau = gen_constant_signal(
            cfg.n_train + cfg.n_test,
            p,
            cfg.rct_probability,
            rng
        )

        # Generate validation data to select lambda of shrinkage estimators
        Y_val, X_val_curr, D_val, Y0_val, Y1_val, _ = gen_constant_signal(
            cfg.n_val,
            p,
            cfg.rct_probability,
            rng
        )

        # Train test split
        Y_train, X_train_curr, D_train, Y0_train, Y1_train, tau_train, \
            Y_test, X_test_curr, D_test, Y0_test, Y1_test, tau_test = split_train_test(cfg.n_train, Y, X, D, Y0, Y1, tau)

        # Create the classification weights and target
        weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)
        _, target_test = make_weight_and_target(Y_test, D_test, cfg.rct_probability)

        # Estimate param
        alpha_0 = rng.random((X_train_curr.shape[1], 1))
        alpha_hat = gd(alpha_0, X_train_curr, target_train, weight, lr = cfg.lr, steps = cfg.steps)

        # Compute welfare
        train_welfare[idx] = compute_welfare(X_train_curr @ alpha_hat, Y0_train, Y1_train)
        test_welfare[idx] = compute_welfare(X_test_curr @ alpha_hat, Y0_test, Y1_test)
        oracle_train_welfare[idx] = compute_welfare(tau_train, Y0_train, Y1_train)
        oracle_test_welfare[idx] = compute_welfare(tau_test, Y0_test, Y1_test)

        # Compute training loss
        train_loss[idx] = _logistic_loss(X_train_curr, target_train, alpha_hat, weight)

        # Compute classification risk
        train_risk[idx] = np.mean( np.sign(X_train_curr @ alpha_hat) != target_train )
        test_risk[idx] = np.mean( np.sign(X_test_curr @ alpha_hat) != target_test )

        # Compute norm of coefficient
        beta_norm[idx] = np.linalg.norm(alpha_hat)

        ### Run SVM
        svc_w_norm[idx] = np.nan
        svc_test_welfare[idx] = np.nan
        if check_interpolation(X_train_curr, target_train):
            svc = LinearSVC(
                C=1e10,  # very large
                loss="hinge",
                fit_intercept = False,
                tol=1e-6,
                max_iter=1000000
            )
            svc.fit(X_train_curr, target_train.ravel())
            w_svc = svc.coef_.ravel()
            svc_w_norm[idx] = np.linalg.norm(w_svc)

            # SVC welfare
            svc_test_welfare[idx] = compute_welfare(X_test_curr @ w_svc.reshape(-1,1), Y0_test, Y1_test)


        ### Run Ridge
        ridge = ShrinkageLogistic(l1_ratio = 0)
        lambda_grid = np.logspace(-3, 3, 50)
        ridge.select_lambda(lambda_grid, X_train_curr, target_train.ravel(), weight.ravel(), X_val_curr, Y0_val, Y1_val)
        ridge.fit(X_train_curr, target_train.ravel(), weight.ravel())

        # Compute ridge welfare
        beta = ridge.model_.coef_.reshape(-1,1)
        ridge_test_welfare[idx] = compute_welfare(X_test_curr @ beta, Y0_test, Y1_test)


        ### Run LASSO
        lasso = ShrinkageLogistic(l1_ratio = 1)
        lambda_grid = np.logspace(-4, 4, 20)
        lasso.select_lambda(lambda_grid, X_train_curr, target_train.ravel(), weight.ravel(), X_val_curr, Y0_val, Y1_val)
        lasso.fit(X_train_curr, target_train.ravel(), weight.ravel())

        # Compute ridge welfare
        beta = lasso.model_.coef_.reshape(-1, 1)
        lasso_test_welfare[idx] = compute_welfare(X_test_curr @ beta, Y0_test, Y1_test)

    print(f"Finished iteration {trial_idx}.")

    return (
        train_welfare, test_welfare, oracle_train_welfare, oracle_test_welfare,
        train_loss, train_risk, test_risk, beta_norm,
        svc_test_welfare, svc_w_norm, ridge_test_welfare, lasso_test_welfare,
        p_grid
    )



if __name__ == '__main__':
    main()


