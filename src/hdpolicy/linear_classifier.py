import numpy as np

def _logistic_loss_grad(X, y, beta, w = None):
    if w is None:
        w = np.ones_like(y)

    return np.mean(w * (1 / (1 + np.exp(y * X @ beta))) * (- y * X), axis=0).reshape(-1, 1)


def gd(beta_0, X, y, w = None, lr = 1e-2, threshold = 1e-4):
    gap = np.inf
    while gap > threshold:   ### Consider a different stopping condition e.g. gradient norm or relative change
        beta_1 = beta_0 - lr * _logistic_loss_grad(X, y, beta_0, w)
        gap = np.linalg.norm(beta_1 - beta_0)
        beta_0 = beta_1

    return beta_0
