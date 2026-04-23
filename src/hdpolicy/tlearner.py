import numpy as np

def fit_linear_regression(X, y):
    """
    Linear regression via Moore–Penrose pseudoinverse.

    - If n >= p: OLS solution
    - If n < p: minimum-norm interpolator

    Includes intercept.
    """
    coef = np.linalg.pinv(X) @ y
    return coef


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



def predict_linear_regression(X, coef):

    return X @ coef


