import numpy as np

def split_train_test(n_train: int, *arrays):

    train = [arr[:n_train] for arr in arrays]
    test = [arr[n_train:] for arr in arrays]

    return (*train, *test)


def make_weight_and_target(Y, D, rct_prob):

    weight = np.abs(Y) / ( D*rct_prob + (1-D)/2 )
    target = np.sign(Y) * D

    return (weight, target)


def check_interpolation(X, y, atol = 0.02):

    w_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.where(X @ w_ls >= 0, 1, -1)
    err = np.mean(pred != y)

    return np.isclose(err, 0, atol = atol)


def standardize_train_test(Phi_train, Phi_test):

    mean = Phi_train.mean(axis=0, keepdims=True)
    std = Phi_train.std(axis=0, keepdims=True) + 1e-10

    Phi_train_std = (Phi_train - mean) / std
    Phi_test_std = (Phi_test - mean) / std

    return Phi_train_std, Phi_test_std


def add_intercept(Phi):
    n = Phi.shape[0]
    return np.hstack([np.ones((n, 1)), Phi])

