import numpy as np

from tqdm import tqdm

from hdpolicy.config import Config
from hdpolicy.rng import make_rng
from hdpolicy.gen_data import gen_rct
from hdpolicy.linear_classifier import gd
from hdpolicy.metrics import compute_welfare

def run_simple_linear(cfg: Config):

    rng = make_rng(cfg.seed)

    p_grid = np.arange(5, cfg.dim_max + 1, 10)

    train_welfare = np.zeros(len(p_grid))
    test_welfare = np.zeros(len(p_grid))

    beta_0 = rng.normal(-1, 2, size = (cfg.dim_max, 1))
    beta_1 = rng.normal(1, 2, size = (cfg.dim_max, 1))

    for  _ in tqdm(range(cfg.n_rep)):

        for idx in range(len(p_grid)):

            # Generate data
            p = p_grid[idx]
            print(p)

            beta_0_curr, beta_1_curr = beta_0[:p], beta_1[:p]
            # beta_0_curr = beta_0_curr / np.linalg.norm(beta_0_curr)
            # beta_1_curr = beta_1_curr / np.linalg.norm(beta_1_curr)

            Y, X, D, Y0, Y1 = gen_rct(cfg.n_train + cfg.n_test, p, cfg.rct_probability, rng, beta_0=beta_0_curr,
                                      beta_1=beta_1_curr)

            # Train test split
            X_train, X_test = X[:cfg.n_train], X[cfg.n_train:]
            Y_train, D_train = Y[:cfg.n_train], D[:cfg.n_train]

            Y0_train, Y1_train = Y0[:cfg.n_train], Y1[:cfg.n_train]
            Y0_test, Y1_test = Y0[cfg.n_train:], Y1[cfg.n_train:]

            target_train = np.sign(Y_train) * D_train
            weight = np.abs(Y_train) / (D_train * cfg.rct_probability + (1 - D_train) / 2)


            # Compute param
            alpha_0 = rng.random((p, 1))
            alpha_hat = gd(alpha_0, X_train, target_train, weight)

            # Compute welfare
            train_welfare[idx] += compute_welfare(X_train, alpha_hat, Y0_train, Y1_train)
            test_welfare[idx] += compute_welfare(X_test[:, :p], alpha_hat, Y0_test, Y1_test)


    train_welfare /= cfg.n_rep
    test_welfare /= cfg.n_rep

    return {
        "train_welfare": train_welfare,
        "test_welfare": test_welfare,
        "p_grid": p_grid
    }


