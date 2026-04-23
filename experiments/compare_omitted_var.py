import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


# ============================================================
# 1. Data generating process
# ============================================================

def gen_rct(
    n,
    n_dim,
    rct_prob,
    rng,
    beta_0=None,
    beta_1=None,
    noise_std=1.0,
):
    """
    Generate RCT data with D in {-1, +1}.

    Y0 = X beta_0 + eps0
    Y1 = X beta_1 + eps1
    D  ~ Bernoulli(rct_prob), coded as +1 / -1
    Y  = observed outcome
    tau = true CATE = X (beta_1 - beta_0)

    Returns
    -------
    Y  : (n,)
    X  : (n, n_dim)
    D  : (n,)
    Y0 : (n,)
    Y1 : (n,)
    tau: (n,)
    beta_0, beta_1 : (n_dim,)
    """
    if beta_0 is None:
        beta_0 = rng.normal(-1.0, 1.0, size=n_dim)
    if beta_1 is None:
        beta_1 = rng.normal(1.0, 1.0, size=n_dim)

    X = rng.normal(0.0, 1.0, size=(n, n_dim))

    eps0 = rng.normal(0.0, noise_std, size=n)
    eps1 = rng.normal(0.0, noise_std, size=n)

    Y0 = X @ beta_0 + eps0
    Y1 = X @ beta_1 + eps1

    D = rng.choice([1, -1], size=n, p=[rct_prob, 1 - rct_prob])
    Y = ((1 + D) / 2.0) * Y1 + ((1 - D) / 2.0) * Y0

    tau = X @ (beta_1 - beta_0)

    return Y, X, D, Y0, Y1, tau, beta_0, beta_1


# ============================================================
# 2. Helpers: linear regression
# ============================================================

def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def fit_linear_regression(X, y):
    """
    Linear regression via Moore–Penrose pseudoinverse.

    - If n >= p: OLS solution
    - If n < p: minimum-norm interpolator

    Includes intercept.
    """
    # X1 = add_intercept(X)  # (n, p+1)
    X1 = X
    coef = np.linalg.pinv(X1) @ y
    return coef


def predict_linear_regression(X, coef):
    # X1 = add_intercept(X)
    X1 = X

    return X1 @ coef


# ============================================================
# 3. T-learner for CATE
# ============================================================

def fit_t_learner_linear(X, D, Y, treat_label=1, control_label=-1, ridge=1e-8):
    """
    Fit separate linear regressions:
      mu1(x) = E[Y | D=+1, X=x]
      mu0(x) = E[Y | D=-1, X=x]
    """
    mask_t = (D == treat_label)
    mask_c = (D == control_label)

    coef_t = fit_linear_regression(X[mask_t], Y[mask_t])
    coef_c = fit_linear_regression(X[mask_c], Y[mask_c])

    return coef_t, coef_c


def predict_cate_t_learner(X, coef_t, coef_c):
    mu1_hat = predict_linear_regression(X, coef_t)
    mu0_hat = predict_linear_regression(X, coef_c)
    return mu1_hat - mu0_hat


def policy_from_cate(tau_hat):
    """
    Policy in {-1, +1}: treat if predicted CATE > 0.
    """
    return np.where(tau_hat > 0, 1, -1)


# ============================================================
# 4. Direct weighted risk minimization
# ============================================================

def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def fit_weighted_logistic_gd(
    X,
    labels,
    sample_weight,
    lr=0.05,
    n_iter=10000,
    l2=1e-4,
    verbose=False,
):
    """
    Weighted logistic regression by gradient descent.

    labels must be in {-1, +1}
    objective:
        mean_i w_i * log(1 + exp(-y_i * (b + x_i'w))) + 0.5 * l2 * ||w||^2
    """
    X1 = add_intercept(X)
    n, p = X1.shape
    theta = np.zeros(p)

    for it in range(n_iter):
        margin = labels * (X1 @ theta)              # shape (n,)
        prob = sigmoid(-margin)                     # = 1 / (1 + exp(margin))

        # gradient of weighted logistic loss
        grad = -(X1.T @ (sample_weight * labels * prob)) / n

        # no penalty on intercept
        reg = l2 * theta
        reg[0] = 0.0
        grad = grad + reg

        theta -= lr * grad

        if verbose and (it % 1000 == 0 or it == n_iter - 1):
            loss = np.mean(sample_weight * np.log1p(np.exp(-margin))) + 0.5 * l2 * np.sum(theta[1:] ** 2)
            print(f"iter={it:5d}, loss={loss:.6f}")

    return theta


def predict_policy_linear_classifier(X, theta):
    score = add_intercept(X) @ theta
    return np.where(score > 0, 1, -1), score


def fit_direct_policy_linear(
    X,
    D,
    Y,
    rct_prob,
    lr=0.05,
    n_iter=5000,
    l2=1e-4,
    verbose=False,
):
    """
    Direct policy learning via weighted classification.

    For RCT with D in {-1,+1}, define
        Gamma_i = D_i * Y_i / p(D_i)
    where p(D_i)=rct_prob if D_i=+1, else 1-rct_prob.

    Since E[Gamma | X] = tau(X), maximizing welfare reduces to
    weighted classification with:
        label_i  = sign(Gamma_i)
        weight_i = |Gamma_i|
    """
    pD = np.where(D == 1, rct_prob, 1.0 - rct_prob)
    gamma = D * Y / pD

    labels = np.where(gamma >= 0, 1, -1)
    weights = np.abs(gamma)

    theta = fit_weighted_logistic_gd(
        X=X,
        labels=labels,
        sample_weight=weights,
        lr=lr,
        n_iter=n_iter,
        l2=l2,
        verbose=verbose,
    )
    return theta, gamma


# ============================================================
# 5. Welfare evaluation
# ============================================================

def policy_value(policy, Y0, Y1):
    """
    True welfare under policy in {-1, +1}.
    """
    return np.mean(np.where(policy == 1, Y1, Y0))


def oracle_policy_from_tau(tau):
    return np.where(tau > 0, 1, -1)


def treatment_accuracy(policy_hat, policy_oracle):
    return np.mean(policy_hat == policy_oracle)


# ============================================================
# 6. One replication over a dimension grid
# ============================================================

def run_one_replication(
    n_train=1000,
    n_test=5000,
    n_dim_max=100,
    dim_grid=None,
    rct_prob=0.5,
    seed=123,
    noise_std=1.0,
    ridge=1e-8,
    gd_lr=0.05,
    gd_iter=4000,
    gd_l2=1e-4,
):
    if dim_grid is None:
        dim_grid = list(range(5, 101, 5))

    rng = np.random.default_rng(seed)

    # Fix the DGP coefficients within a replication
    beta_0 = rng.normal(-1.0, 2.0, size=n_dim_max)
    beta_1 = rng.normal(1.0, 2.0, size=n_dim_max)

    # Generate train/test once at max dimension
    Y_tr, X_tr, D_tr, Y0_tr, Y1_tr, tau_tr, _, _ = gen_rct(
        n=n_train,
        n_dim=n_dim_max,
        rct_prob=rct_prob,
        rng=rng,
        beta_0=beta_0,
        beta_1=beta_1,
        noise_std=noise_std,
    )

    Y_te, X_te, D_te, Y0_te, Y1_te, tau_te, _, _ = gen_rct(
        n=n_test,
        n_dim=n_dim_max,
        rct_prob=rct_prob,
        rng=rng,
        beta_0=beta_0,
        beta_1=beta_1,
        noise_std=noise_std,
    )

    oracle_policy = oracle_policy_from_tau(tau_te)
    oracle_welfare = policy_value(oracle_policy, Y0_te, Y1_te)

    rows = []

    for k in tqdm(dim_grid):
        Xtr_k = X_tr[:, :k]
        Xte_k = X_te[:, :k]

        # ---------- Approach 1: T-learner -> plug-in policy ----------
        coef_t, coef_c = fit_t_learner_linear(Xtr_k, D_tr, Y_tr, ridge=ridge)
        tau_hat_te = predict_cate_t_learner(Xte_k, coef_t, coef_c)
        policy_t = policy_from_cate(tau_hat_te)

        welfare_t = policy_value(policy_t, Y0_te, Y1_te)
        regret_t = oracle_welfare - welfare_t
        acc_t = treatment_accuracy(policy_t, oracle_policy)
        cate_mse_t = np.mean((tau_hat_te - tau_te) ** 2)

        # ---------- Approach 2: direct weighted risk minimization ----------
        theta_direct, gamma_tr = fit_direct_policy_linear(
            X=Xtr_k,
            D=D_tr,
            Y=Y_tr,
            rct_prob=rct_prob,
            lr=gd_lr,
            n_iter=gd_iter,
            l2=gd_l2,
            verbose=False,
        )
        policy_d, score_d = predict_policy_linear_classifier(Xte_k, theta_direct)

        welfare_d = policy_value(policy_d, Y0_te, Y1_te)
        regret_d = oracle_welfare - welfare_d
        acc_d = treatment_accuracy(policy_d, oracle_policy)


        # Get training accuracy and welfare
        policy_train, score_train = predict_policy_linear_classifier(Xtr_k, theta_direct)

        welfare_train = policy_value(policy_train, Y0_tr, Y1_tr)
        regret_train = oracle_welfare - welfare_train
        acc_train = treatment_accuracy(policy_train, D_tr * np.sign(Y_tr))


        rows.append({
            "dim": k,
            "oracle_welfare": oracle_welfare,

            "welfare_tlearner": welfare_t,
            "regret_tlearner": regret_t,
            "acc_tlearner": acc_t,
            "cate_mse_tlearner": cate_mse_t,

            "welfare_direct": welfare_d,
            "regret_direct": regret_d,
            "acc_direct": acc_d,

            "welfare_train": welfare_train,
            "acc_train": acc_train
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. Multiple replications
# ============================================================

def run_simulation(
    n_rep=50,
    n_train=1000,
    n_test=5000,
    n_dim_max=100,
    dim_grid=None,
    rct_prob=0.5,
    base_seed=123,
    noise_std=1.0,
    ridge=1e-8,
    gd_lr=0.05,
    gd_iter=4000,
    gd_l2=1e-4,
):
    if dim_grid is None:
        dim_grid = list(range(5, 101, 5))

    all_dfs = []
    for rep in range(n_rep):
        df_rep = run_one_replication(
            n_train=n_train,
            n_test=n_test,
            n_dim_max=n_dim_max,
            dim_grid=dim_grid,
            rct_prob=rct_prob,
            seed=base_seed + rep,
            noise_std=noise_std,
            ridge=ridge,
            gd_lr=gd_lr,
            gd_iter=gd_iter,
            gd_l2=gd_l2,
        )
        df_rep["rep"] = rep
        all_dfs.append(df_rep)

    out = pd.concat(all_dfs, ignore_index=True)
    return out


def summarize_results(df):
    summary = (
        df.groupby("dim")
        .agg(
            welfare_tlearner_mean=("welfare_tlearner", "mean"),
            welfare_tlearner_sd=("welfare_tlearner", "std"),
            welfare_direct_mean=("welfare_direct", "mean"),
            welfare_direct_sd=("welfare_direct", "std"),

            regret_tlearner_mean=("regret_tlearner", "mean"),
            regret_direct_mean=("regret_direct", "mean"),

            acc_tlearner_mean=("acc_tlearner", "mean"),
            acc_direct_mean=("acc_direct", "mean"),

            cate_mse_tlearner_mean=("cate_mse_tlearner", "mean"),
            oracle_welfare_mean=("oracle_welfare", "mean"),

            welfare_train_mean=("welfare_train", "mean"),
            acc_train_mean=("acc_train", "mean")
        )
        .reset_index()
    )
    return summary


# ============================================================
# 8. Plotting
# ============================================================

def plot_welfare(summary_df):
    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["dim"], summary_df["welfare_tlearner_mean"], marker="o", label="T-learner plug-in")
    plt.plot(summary_df["dim"], summary_df["welfare_direct_mean"], marker="s", label="Direct weighted classification")
    plt.plot(summary_df["dim"], summary_df["oracle_welfare_mean"], linestyle="--", label="Oracle policy")
    plt.plot(summary_df["dim"], summary_df["welfare_train_mean"], marker="s", label="Training welfare")
    plt.xlabel("Observed dimension")
    plt.ylabel("True test welfare")
    plt.title("Policy welfare vs observed dimension")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regret(summary_df):
    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["dim"], summary_df["regret_tlearner_mean"], marker="o", label="T-learner regret")
    plt.plot(summary_df["dim"], summary_df["regret_direct_mean"], marker="s", label="Direct regret")
    plt.xlabel("Observed dimension")
    plt.ylabel("Oracle welfare - learned policy welfare")
    plt.title("Policy regret vs observed dimension")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy(summary_df):
    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["dim"], summary_df["acc_tlearner_mean"], marker="o", label="T-learner assignment accuracy")
    plt.plot(summary_df["dim"], summary_df["acc_direct_mean"], marker="s", label="Direct assignment accuracy")

    plt.plot(summary_df["dim"], summary_df["acc_train_mean"], marker="s", label="Training accuracy")

    plt.xlabel("Observed dimension")
    plt.ylabel("Agreement with oracle policy")
    plt.title("Treatment assignment accuracy vs observed dimension")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. Example run
# ============================================================

if __name__ == "__main__":
    dim_grid = list(range(5, 301, 5))

    df = run_simulation(
        n_rep=5,
        n_train=150,
        n_test=5000,
        n_dim_max=300,
        dim_grid=dim_grid,
        rct_prob=0.5,
        base_seed=123,
        noise_std=1.0,
        ridge=0,
        gd_lr=0.05,
        gd_iter=20000,
        gd_l2=0
    )

    summary = summarize_results(df)
    print(summary.tail(10))

    plot_welfare(summary)
    plot_regret(summary)
    plot_accuracy(summary)




