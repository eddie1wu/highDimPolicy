from pathlib import Path

import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_rct
from hdpolicy.utils import *
from hdpolicy.linear_classifier import gd, _logistic_loss
from hdpolicy.tlearner import *
from hdpolicy.metrics import compute_welfare

from hdpolicy.io.save_load import load_config, save_config, make_run_dir, save_results



def main():

    cfg = load_config("configs/tlearner_ReLU.toml")

    results = run_tlearner_ReLU(cfg)

    run_dir = make_run_dir(Path("results"), tag="tlearner_ReLU")
    save_results(run_dir, results)
    save_config(run_dir, cfg)

    print(f"Saved results to {run_dir}.")



def run_tlearner_ReLU(cfg, max_workers: int | None = None):

    max_workers = max_workers if max_workers is not None else os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(executor.map( run_single_trial, itertools.repeat(cfg), range(cfg.n_rep) ))

    # Unpack results
    results_list = zip(*results_list)

    # Average over reps
    results_list = [np.nanmean(r, axis = 0) for r in results_list]

    tl_train_welfare, tl_test_welfare, clf_train_welfare, clf_test_welfare, \
        tl_original_X_welfare, clf_original_X_welfare, oracle_train_welfare, oracle_test_welfare, \
        clf_train_loss, clf_train_risk, clf_test_risk, \
        p_grid = results_list

    return {
        'tl_train_welfare': tl_train_welfare,
        'tl_test_welfare': tl_test_welfare,
        'clf_train_welfare': clf_train_welfare,
        'clf_test_welfare': clf_test_welfare,
        'tl_original_X_welfare': tl_original_X_welfare,
        'clf_original_X_welfare': clf_original_X_welfare,
        'oracle_train_welfare': oracle_train_welfare,
        'oracle_test_welfare': oracle_test_welfare,
        'clf_train_loss': clf_train_loss,
        'clf_train_risk': clf_train_risk,
        'clf_test_risk': clf_test_risk,
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

    # Generate random neural features
    W = rng.normal(0, 1 / np.sqrt(X.shape[1]), size=(cfg.X_dim_max, cfg.dim_max))
    Phi = np.maximum(0, X @ W)

    # Train test split
    Y_train, X_train, Phi_train, D_train, Y0_train, Y1_train, tau_train, \
        Y_test, X_test, Phi_test, D_test, Y0_test, Y1_test, tau_test = split_train_test(cfg.n_train, Y, X, Phi, D, Y0, Y1, tau)

    # Create the classification weights and target
    weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)
    _, target_test = make_weight_and_target(Y_test, D_test, cfg.rct_probability)

    # Create feature grid
    p_grid = np.arange(5, cfg.dim_max + 1, 10)

    # Store outputs
    tl_train_welfare = np.zeros(len(p_grid))
    tl_test_welfare = np.zeros(len(p_grid))
    clf_train_welfare = np.zeros(len(p_grid))
    clf_test_welfare = np.zeros(len(p_grid))
    tl_original_X_welfare = 0
    clf_original_X_welfare = 0
    oracle_train_welfare = 0
    oracle_test_welfare = 0
    clf_train_loss = np.zeros(len(p_grid))
    clf_train_risk = np.zeros(len(p_grid))
    clf_test_risk = np.zeros(len(p_grid))


    for idx in tqdm(range(len(p_grid))):

        p = p_grid[idx]

        Phi_train_curr, Phi_test_curr = Phi_train[:, :p], Phi_test[:, :p]
        Phi_train_curr = add_intercept(Phi_train_curr)
        Phi_test_curr = add_intercept(Phi_test_curr)

        ### Two learner
        coef_t, coef_c = fit_t_learner_linear(Phi_train_curr, D_train.ravel(), Y_train)
        tau_hat_tr = predict_cate_t_learner(Phi_train_curr, coef_t, coef_c)
        tau_hat_te = predict_cate_t_learner(Phi_test_curr, coef_t, coef_c)

        tl_train_welfare[idx] = compute_welfare(tau_hat_tr, Y0_train, Y1_train)
        tl_test_welfare[idx] = compute_welfare(tau_hat_te, Y0_test, Y1_test)


        ### Weighted classifier
        alpha_0 = rng.random((Phi_train_curr.shape[1], 1))
        alpha_hat = gd(alpha_0, Phi_train_curr, target_train, weight, lr = cfg.lr, steps = cfg.steps)

        # Compute welfare
        clf_train_welfare[idx] = compute_welfare(Phi_train_curr @ alpha_hat, Y0_train, Y1_train)
        clf_test_welfare[idx] = compute_welfare(Phi_test_curr @ alpha_hat, Y0_test, Y1_test)

        # Compute training loss
        clf_train_loss[idx] = _logistic_loss(Phi_train_curr, target_train, alpha_hat, weight)

        # Compute classification risk
        clf_train_risk[idx] = np.mean( np.sign(Phi_train_curr @ alpha_hat) != target_train )
        clf_test_risk[idx] = np.mean( np.sign(Phi_test_curr @ alpha_hat) != target_test )


    ### Two learner full feature welfare
    X_train_curr = add_intercept(X_train)
    X_test_curr = add_intercept(X_test)
    coef_t, coef_c = fit_t_learner_linear(X_train_curr, D_train.ravel(), Y_train)

    tau_hat_te = predict_cate_t_learner(X_test_curr, coef_t, coef_c)
    tl_original_X_welfare += compute_welfare(tau_hat_te, Y0_test, Y1_test)


    ### Classifier full feature welfare
    alpha_0 = rng.random((X_train_curr.shape[1], 1))
    alpha_hat = gd(alpha_0, X_train_curr, target_train, weight, lr = cfg.lr, steps = cfg.steps)
    clf_original_X_welfare += compute_welfare(X_test_curr @ alpha_hat, Y0_test, Y1_test)

    # Oracle welfare
    oracle_train_welfare += compute_welfare(tau_train, Y0_train, Y1_train)
    oracle_test_welfare += compute_welfare(tau_test, Y0_test, Y1_test)

    print(f"Finished iteration {trial_idx}.")

    return (
        tl_train_welfare, tl_test_welfare, clf_train_welfare, clf_test_welfare,
        tl_original_X_welfare, clf_original_X_welfare, oracle_train_welfare, oracle_test_welfare,
        clf_train_loss, clf_train_risk, clf_test_risk,
        p_grid
    )



if __name__ == '__main__':
    main()
