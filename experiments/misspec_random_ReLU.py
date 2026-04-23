from pathlib import Path

import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_polynomial
from hdpolicy.utils import *
from hdpolicy.linear_classifier import gd, _logistic_loss
from hdpolicy.metrics import compute_welfare

from hdpolicy.io.save_load import load_config, save_config, make_run_dir, save_results



def main():

    cfg = load_config("configs/misspec_random_ReLU.toml")

    results = run_random_ReLU(cfg)

    run_dir = make_run_dir(Path("results"), tag="misspec_random_ReLU")
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

    # Create random ReLU sieves
    W = rng.normal( 0, 1 / np.sqrt(cfg.X_dim_max), size=(cfg.X_dim_max, cfg.dim_max) )
    b = rng.normal( 0, 1, size=(cfg.dim_max, ) )
    Phi = np.maximum(0.0, X @ W + b)

    # Traint test split
    Y_train, X_train, Phi_train, D_train, Y0_train, Y1_train, tau_train, \
        Y_test, X_test, Phi_test, D_test, Y0_test, Y1_test, tau_test = split_train_test(cfg.n_train, Y, X, Phi, D, Y0, Y1, tau)

    # Create classification weight and target
    weight, target_train = make_weight_and_target(Y_train, D_train, cfg.rct_probability)
    _, target_test = make_weight_and_target(Y_test, D_test, cfg.rct_probability)

    # Define grid
    p_grid = np.concatenate([
        np.arange(1, 121, 5),
        np.arange(130, cfg.dim_max + 1, 100)
    ])

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


    for idx in tqdm(range(len(p_grid))):

        p = p_grid[idx]

        Phi_train_curr, Phi_test_curr = Phi_train[:, :p], Phi_test[:, :p]

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





#
#     # constant-treat baselines
#     always_treat_train = np.ones_like(Y0_train)
#     always_treat_test = np.ones_like(Y0_test)
#     never_treat_train = np.zeros_like(Y0_train)
#     never_treat_test = np.zeros_like(Y0_test)
#
#     baseline = {
#         "always_train": compute_welfare(always_treat_train, Y0_train, Y1_train),
#         "always_test": compute_welfare(always_treat_test, Y0_test, Y1_test),
#         "never_train": compute_welfare(never_treat_train, Y0_train, Y1_train),
#         "never_test": compute_welfare(never_treat_test, Y0_test, Y1_test),
#     }
#
#     return (
#         ms,
#         np.array(train_welfares),
#         np.array(test_welfares),
#         np.array(train_losses),
#         np.array(oracle_train_welfares),
#         np.array(oracle_test_welfares),
#         baseline,
#         n_train,
#         np.array(train_risk),
#         np.array(test_risk)
#     )
#
#
#
#
#
#
#
#
# # -----------------------------
# # Stable sigmoid
# # -----------------------------
# def sigmoid(z):
#     out = np.empty_like(z)
#     pos = z >= 0
#     neg = ~pos
#     out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
#     ez = np.exp(z[neg])
#     out[neg] = ez / (1.0 + ez)
#     return out
#
#
# # -----------------------------
# # Weighted logistic loss
# # y in {-1, +1}
# # weight >= 0
# # -----------------------------
# def weighted_logistic_loss(Phi, y, w, weight):
#     margin = y * (Phi @ w)
#     return np.mean(weight * np.log1p(np.exp(-margin)))
#
#
# # -----------------------------
# # Policy DGP
# #
# # tau(x) is the same nonlinear signal as your
# # old logistic decision boundary:
# #   tau(x) = x1 + x1^2 + x2 + x2^2
# #
# # Potential outcomes:
# #   Y(0) = base(x) - tau(x)/2 + eps0
# #   Y(1) = base(x) + tau(x)/2 + eps1
# #
# # So E[Y(1)-Y(0)|X=x] = tau(x)
# # -----------------------------
# def gen_policy_data(n, rng, treat_prob=0.5, noise_std=1.0):
#     X = rng.normal(0, 1, size=(n, 2))
#
#     x1 = X[:, 0:1]
#     x2 = X[:, 1:2]
#
#     tau = x1 + x1**2 + x2 + x2**2
#
#     # baseline outcome level
#     base = 0.5 * x1 - 0.5 * x2
#
#     eps0 = noise_std * rng.normal(size=(n, 1))
#     eps1 = noise_std * rng.normal(size=(n, 1))
#
#     Y0 = base - 0.5 * tau + eps0
#     Y1 = base + 0.5 * tau + eps1
#
#     D = rng.binomial(1, treat_prob, size=(n, 1)).astype(float)
#     Y = D * Y1 + (1.0 - D) * Y0
#
#     return X, D, Y, Y0, Y1, tau
#
#
# # -----------------------------
# # Random ReLU feature dictionary
# # phi_j(x) = ReLU(a_j x + b_j)
# # -----------------------------
# def relu_features(X, a, b):
#     return np.maximum(0.0, X @ a.T + b)
#
#
# # -----------------------------
# # Add intercept
# # -----------------------------
# def add_intercept(Phi):
#     n = Phi.shape[0]
#     return np.hstack([np.ones((n, 1)), Phi])
#
#
# # -----------------------------
# # Standardize using training stats only
# # Do not standardize intercept column
# # -----------------------------
# def standardize_train_test(Phi_train, Phi_test, intercept=True):
#     if intercept:
#         Z_train = Phi_train[:, 1:]
#         Z_test = Phi_test[:, 1:]
#
#         mean = Z_train.mean(axis=0, keepdims=True)
#         std = Z_train.std(axis=0, keepdims=True) + 1e-8
#
#         Z_train = (Z_train - mean) / std
#         Z_test = (Z_test - mean) / std
#
#         Phi_train_std = np.hstack([Phi_train[:, :1], Z_train])
#         Phi_test_std = np.hstack([Phi_test[:, :1], Z_test])
#     else:
#         mean = Phi_train.mean(axis=0, keepdims=True)
#         std = Phi_train.std(axis=0, keepdims=True) + 1e-8
#
#         Phi_train_std = (Phi_train - mean) / std
#         Phi_test_std = (Phi_test - mean) / std
#
#     return Phi_train_std, Phi_test_std
#
#
# # -----------------------------
# # Build policy-learning target from RCT data
# #
# # Gamma = Y * (D - e) / (e(1-e))
# #
# # Then solve weighted classification with:
# #   target = sign(Gamma) in {-1,+1}
# #   weight = |Gamma|
# # -----------------------------
# def make_policy_weight_and_target(Y, D, e):
#     gamma = Y * (D - e) / (e * (1.0 - e))
#     target = np.where(gamma >= 0.0, 1.0, -1.0)
#     weight = np.abs(gamma)
#     return weight, target
#
#
# # -----------------------------
# # Weighted GD for logistic loss
# # -----------------------------
# def train_weighted_logistic_gd(Phi, y, sample_weight, lr=0.05, steps=20000, return_history=False):
#     n, p = Phi.shape
#     w = np.zeros((p, 1))
#
#     if return_history:
#         losses = []
#         norms = []
#
#     for _ in range(steps):
#         margin = y * (Phi @ w)  # (n,1)
#         # derivative of weight_i * log(1 + exp(-margin_i))
#         grad = -(Phi.T @ (sample_weight * y * sigmoid(-margin))) / n
#         w -= lr * grad
#
#         if return_history:
#             losses.append(weighted_logistic_loss(Phi, y, w, sample_weight))
#             norms.append(np.linalg.norm(w))
#
#     if return_history:
#         return w, {
#             "loss": np.array(losses),
#             "norm": np.array(norms),
#         }
#     return w
#
#
# # -----------------------------
# # Policy prediction
# # policy = 1 if score >= 0 else 0
# # -----------------------------
# def predict_policy(Phi, w):
#     scores = Phi @ w
#     return (scores >= 0).astype(float)
#
#
# # -----------------------------
# # Welfare:
# # W(pi) = mean[ pi(X) * Y1 + (1-pi(X)) * Y0 ]
# # -----------------------------
# def compute_welfare(policy, Y0, Y1):
#     return np.mean(policy * Y1 + (1.0 - policy) * Y0)
#
#
# # -----------------------------
# # Oracle policy from true tau
# # -----------------------------
# def oracle_policy_from_tau(tau):
#     return (tau >= 0).astype(float)
#
#
# # -----------------------------
# # One nested random-ReLU sieve experiment
# # -----------------------------
# def run_experiment_random_relu_policy(seed):
#     rng = np.random.default_rng(seed)
#
#     n_train = 100
#     n_test = 10000
#     M_max = 1200
#     treat_prob = 0.5
#
#     X_train, D_train, Y_train, Y0_train, Y1_train, tau_train = gen_policy_data(
#         n_train, rng, treat_prob=treat_prob, noise_std=1.0
#     )
#     X_test, D_test, Y_test, Y0_test, Y1_test, tau_test = gen_policy_data(
#         n_test, rng, treat_prob=treat_prob, noise_std=1.0
#     )
#
#     # one large random dictionary -> nested sieve
#     a_full = rng.normal(0, 1, size=(M_max, X_train.shape[1]))
#     b_full = rng.normal(0, 1, size=(M_max,))
#
#     # ms = np.arange(5, M_max + 1, 200)
#     ms = np.concatenate([
#         np.arange(1, 111, 5),
#         np.arange(210, M_max+1, 100)
#     ])
#
#     train_welfares = []
#     test_welfares = []
#     train_losses = []
#     oracle_train_welfares = []
#     oracle_test_welfares = []
#     train_risk = []
#     test_risk = []
#
#     # build policy-learning labels from observed RCT data
#     sample_weight, target = make_policy_weight_and_target(Y_train, D_train, treat_prob)
#
#     for m in tqdm(ms):
#         Phi_train = relu_features(X_train, a_full[:m], b_full[:m])
#         Phi_test = relu_features(X_test, a_full[:m], b_full[:m])
#
#         Phi_train = add_intercept(Phi_train)
#         Phi_test = add_intercept(Phi_test)
#
#         Phi_train, Phi_test = standardize_train_test(Phi_train, Phi_test, intercept=True)
#
#         ### Check interpolation
#         w_ls, *_ = np.linalg.lstsq(Phi_train, target, rcond = None)
#         pred = np.where(Phi_train @ w_ls >= 0, 1, -1)
#         print("training error:", np.mean(pred != target))
#
#         w = train_weighted_logistic_gd(
#             Phi_train,
#             target,
#             sample_weight,
#             lr=0.5,
#             steps=300000
#         )
#
#         pi_train = predict_policy(Phi_train, w)
#         pi_test = predict_policy(Phi_test, w)
#
#         train_welfares.append(compute_welfare(pi_train, Y0_train, Y1_train))
#         test_welfares.append(compute_welfare(pi_test, Y0_test, Y1_test))
#         train_losses.append(weighted_logistic_loss(Phi_train, target, w, sample_weight))
#
#         # same oracle repeated across m, just for reference
#         pi_oracle_train = oracle_policy_from_tau(tau_train)
#         pi_oracle_test = oracle_policy_from_tau(tau_test)
#
#         oracle_train_welfares.append(compute_welfare(pi_oracle_train, Y0_train, Y1_train))
#         oracle_test_welfares.append(compute_welfare(pi_oracle_test, Y0_test, Y1_test))
#
#         # Compute risk to monitor interpolation
#         _, target_test = make_policy_weight_and_target(Y_test, D_test, treat_prob)
#         temp = np.mean(np.sign(Phi_train @ w) != target)
#         train_risk.append(temp)
#         temp = np.mean(np.sign(Phi_test @ w) != target_test)
#         test_risk.append(temp)
#
#
#     # constant-treat baselines
#     always_treat_train = np.ones_like(Y0_train)
#     always_treat_test = np.ones_like(Y0_test)
#     never_treat_train = np.zeros_like(Y0_train)
#     never_treat_test = np.zeros_like(Y0_test)
#
#     baseline = {
#         "always_train": compute_welfare(always_treat_train, Y0_train, Y1_train),
#         "always_test": compute_welfare(always_treat_test, Y0_test, Y1_test),
#         "never_train": compute_welfare(never_treat_train, Y0_train, Y1_train),
#         "never_test": compute_welfare(never_treat_test, Y0_test, Y1_test),
#     }
#
#     return (
#         ms,
#         np.array(train_welfares),
#         np.array(test_welfares),
#         np.array(train_losses),
#         np.array(oracle_train_welfares),
#         np.array(oracle_test_welfares),
#         baseline,
#         n_train,
#         np.array(train_risk),
#         np.array(test_risk)
#     )
#
#
# # -----------------------------
# # Run and average
# # -----------------------------
# n_runs = 50
#
# all_train_welfares = []
# all_test_welfares = []
# all_train_losses = []
# all_oracle_train_welfares = []
# all_oracle_test_welfares = []
# all_baselines = []
# all_train_risk = []
# all_test_risk = []
#
# for t in range(n_runs):
#     (
#         ms,
#         train_welfare,
#         test_welfare,
#         train_loss,
#         oracle_train_welfare,
#         oracle_test_welfare,
#         baseline,
#         n_train,
#         train_risk,
#         test_risk
#     ) = run_experiment_random_relu_policy(t)
#
#     all_train_welfares.append(train_welfare)
#     all_test_welfares.append(test_welfare)
#     all_train_losses.append(train_loss)
#     all_oracle_train_welfares.append(oracle_train_welfare)
#     all_oracle_test_welfares.append(oracle_test_welfare)
#     all_baselines.append(baseline)
#     all_train_risk.append(train_risk)
#     all_test_risk.append(test_risk)
#
# train_welfare = np.mean(np.array(all_train_welfares), axis=0)
# test_welfare = np.mean(np.array(all_test_welfares), axis=0)
# train_loss = np.mean(np.array(all_train_losses), axis=0)
# oracle_train_welfare = np.mean(np.array(all_oracle_train_welfares), axis=0)
# oracle_test_welfare = np.mean(np.array(all_oracle_test_welfares), axis=0)
# train_risk = np.mean(np.array(all_train_risk), axis=0)
# test_risk = np.mean(np.array(all_test_risk), axis=0)
#
#
# always_train = np.mean([b["always_train"] for b in all_baselines])
# always_test = np.mean([b["always_test"] for b in all_baselines])
# never_train = np.mean([b["never_train"] for b in all_baselines])
# never_test = np.mean([b["never_test"] for b in all_baselines])
#
#
# # -----------------------------
# # Plot welfare
# # -----------------------------
# plt.figure(figsize=(7, 4.5))
# plt.plot(ms, train_welfare, marker="o", label="Training welfare")
# plt.plot(ms, test_welfare, marker="o", label="Test welfare")
# plt.plot(ms, oracle_test_welfare, linestyle="--", label="Oracle test welfare")
# # plt.axhline(always_test, linestyle=":", label="Always-treat test welfare")
# # plt.axhline(never_test, linestyle=":", label="Never-treat test welfare")
# plt.axvline(n_train, linestyle="--", label="n_train")
# plt.xlabel("Number of random ReLU features")
# plt.ylabel("Welfare")
# plt.title("Policy Learning with Nested Random ReLU Sieve")
# plt.grid(True)
# plt.legend()
# plt.savefig("welfare.png")
#
#
# ### Plot risk
# plt.figure(figsize=(7, 4.5))
# plt.plot(ms, train_risk, marker="o", label="Training risk")
# plt.plot(ms, test_risk, marker="o", label="Test risk")
# plt.axvline(n_train, linestyle="--", label="n_train")
# plt.xlabel("Number of random ReLU features")
# plt.ylabel("Risk")
# plt.title("Policy Learning with Nested Random ReLU Sieve")
# plt.grid(True)
# plt.legend()
# plt.savefig("risk.png")
#
#
# # -----------------------------
# # Plot weighted logistic loss
# # -----------------------------
# plt.figure(figsize=(7, 4.5))
# plt.plot(ms, train_loss, marker="o", label="Training weighted logistic loss")
# plt.axvline(n_train, linestyle="--", label="n_train")
# plt.xlabel("Number of random ReLU features")
# plt.ylabel("Weighted logistic loss")
# plt.title("Training Objective")
# plt.grid(True)
# plt.legend()
# plt.savefig("loss.png")


if __name__ == "__main__":
    main()


