from pathlib import Path

import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from sklearn.svm import LinearSVC

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_rct
from hdpolicy.utils import split_train_test, make_weight_and_target, check_interpolation
from hdpolicy.linear_classifier import gd, _logistic_loss
from hdpolicy.linear_shrinkage import ShrinkageLogistic
from hdpolicy.metrics import compute_welfare

from hdpolicy.io.save_load import load_config, save_config, make_run_dir, save_results

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():

    cfg = load_config("configs/random_ReLU.toml")

    results = run_random_ReLU(cfg)

    run_dir = make_run_dir(Path("results"), tag="random_ReLU")
    save_results(run_dir, results)
    save_config(run_dir, cfg)

    print(f"Saved results to {run_dir}.")



def run_random_ReLU(cfg, max_workers: int | None = None):

    max_workers = max_workers if max_workers is not None else os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(executor.map( run_single_trial, itertools.repeat(cfg), range(cfg.n_rep) ))

    # Unpack results
    results_list = zip(*results_list)

    # Average over reps
    results_list = [np.nanmean(r, axis = 0) for r in results_list]

    train_welfare, test_welfare, original_X_welfare, oracle_train_welfare, oracle_test_welfare, \
        train_loss, train_risk, test_risk, beta_norm, \
        svc_test_welfare, svc_w_norm, ridge_test_welfare, lasso_test_welfare, \
        p_grid = results_list

    return {
        'train_welfare': train_welfare,
        'test_welfare': test_welfare,
        'original_X_welfare': original_X_welfare,
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

    beta_0 = rng.normal(-1, 2, size=(cfg.X_dim_max, 1))
    beta_1 = rng.normal(1, 2, size=(cfg.X_dim_max, 1))

    Y, X, D, Y0, Y1, tau = gen_rct(
        cfg.n_train + cfg.n_test,
        cfg.X_dim_max,
        cfg.rct_probability,
        rng,
        beta_0=beta_0,
        beta_1=beta_1
    )

    # Generate validation data to select lambda of shrinkage estimators
    Y_val, X_val, D_val, Y0_val, Y1_val, _ = gen_rct(
        cfg.n_val,
        cfg.X_dim_max,
        cfg.rct_probability,
        rng,
        beta_0=beta_0,
        beta_1=beta_1
    )

    # Generate random neural features
    W = rng.normal(0, 1 / np.sqrt(X.shape[1]), size=(cfg.X_dim_max, cfg.dim_max))
    Phi = np.maximum(0, X @ W)
    Phi_val = np.maximum(0, X_val @ W)

    # Train test split
    Y_train, X_train, Phi_train, D_train, Y0_train, Y1_train, tau_train, \
        Y_test, X_test, Phi_test, D_test, Y0_test, Y1_test, tau_test = split_train_test(cfg.n_train, Y, X, Phi, D, Y0, Y1, tau)

    # Create the classification weights and target
    weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)
    _, target_test = make_weight_and_target(Y_test, D_test, cfg.rct_probability)

    # Create feature grid
    p_grid = np.arange(5, cfg.dim_max + 1, 10)

    # Store outputs
    train_welfare = np.zeros(len(p_grid))
    test_welfare = np.zeros(len(p_grid))
    original_X_welfare = 0
    oracle_train_welfare = 0
    oracle_test_welfare = 0
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

        Phi_train_curr, Phi_test_curr = Phi_train[:, :p], Phi_test[:, :p]
        # Phi_train_curr, Phi_test_curr = Phi_train[:, :p] / np.sqrt(p) , Phi_test[:, :p] / np.sqrt(p)

        # Estimate param
        alpha_0 = rng.random((p, 1))
        alpha_hat = gd(alpha_0, Phi_train_curr, target_train, weight, lr = cfg.lr, steps = cfg.steps)

        # Compute welfare
        train_welfare[idx] = compute_welfare(Phi_train_curr @ alpha_hat, Y0_train, Y1_train)
        test_welfare[idx] = compute_welfare(Phi_test_curr @ alpha_hat, Y0_test, Y1_test)

        # Compute training loss
        train_loss[idx] = _logistic_loss(Phi_train_curr, target_train, alpha_hat, weight)

        # Compute classification risk
        train_risk[idx] = np.mean( np.sign(Phi_train_curr @ alpha_hat) != target_train )
        test_risk[idx] = np.mean( np.sign(Phi_test_curr @ alpha_hat) != target_test )

        # Compute norm of coefficient
        beta_norm[idx] = np.linalg.norm(alpha_hat)


        ### Run SVM
        svc_w_norm[idx] = np.nan
        svc_test_welfare[idx] = np.nan
        if check_interpolation(Phi_train_curr, target_train):
            svc = LinearSVC(
                C=1e10,  # very large
                loss="hinge",
                fit_intercept = False,
                tol=1e-6,
                max_iter=1000000
            )
            svc.fit(Phi_train_curr, target_train.ravel())
            w_svc = svc.coef_.ravel()
            svc_w_norm[idx] = np.linalg.norm(w_svc)

            # SVC welfare
            svc_test_welfare[idx] = compute_welfare(Phi_test_curr @ w_svc.reshape(-1,1), Y0_test, Y1_test)


        ### Run Ridge
        Phi_val_curr = Phi_val[:, :p]
        # Phi_val_curr = Phi_val[:, :p] / np.sqrt(p)
        ridge = ShrinkageLogistic(l1_ratio = 0)
        lambda_grid = np.logspace(-3, 3, 50)
        ridge.select_lambda(lambda_grid, Phi_train_curr, target_train.ravel(), weight.ravel(), Phi_val_curr, Y0_val, Y1_val)
        ridge.fit(Phi_train_curr, target_train.ravel(), weight.ravel())

        # Compute ridge welfare
        beta = ridge.model_.coef_.reshape(-1,1)
        ridge_test_welfare[idx] = compute_welfare(Phi_test_curr @ beta, Y0_test, Y1_test)


        ### Run LASSO
        lasso = ShrinkageLogistic(l1_ratio = 1)
        lambda_grid = np.logspace(-4, 4, 20)
        lasso.select_lambda(lambda_grid, Phi_train_curr, target_train.ravel(), weight.ravel(), Phi_val_curr, Y0_val, Y1_val)
        lasso.fit(Phi_train_curr, target_train.ravel(), weight.ravel())

        # Compute ridge welfare
        beta = lasso.model_.coef_.reshape(-1, 1)
        lasso_test_welfare[idx] = compute_welfare(Phi_test_curr @ beta, Y0_test, Y1_test)


    # Full feature welfare
    alpha_0 = rng.random((cfg.X_dim_max, 1))
    alpha_hat = gd(alpha_0, X_train, target_train, weight, lr = cfg.lr, steps = cfg.steps)
    original_X_welfare += compute_welfare(X_test @ alpha_hat, Y0_test, Y1_test)

    # Oracle welfare
    oracle_train_welfare += compute_welfare(tau_train, Y0_train, Y1_train)
    oracle_test_welfare += compute_welfare(tau_test, Y0_test, Y1_test)

    print(f"Finished iteration {trial_idx}.")

    return (
        train_welfare, test_welfare, original_X_welfare, oracle_train_welfare, oracle_test_welfare,
        train_loss, train_risk, test_risk, beta_norm,
        svc_test_welfare, svc_w_norm, ridge_test_welfare, lasso_test_welfare,
        p_grid
    )



if __name__ == '__main__':
    main()
