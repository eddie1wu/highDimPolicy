import numpy as np
import matplotlib.pyplot as plt


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)

    return out



def _logistic_loss(X, y, beta, w = None):
    if w is None:
        w = np.ones_like(y)

    margin = y * (X @ beta)

    return np.mean( w * np.log1p( np.exp(-margin) ) )



def _logistic_loss_grad(X, y, beta, w = None):
    if w is None:
        w = np.ones_like(y)

    n, _ = X.shape
    margin = y * (X @ beta)

    return -( X.T @ (w * y * _sigmoid(-margin)) ) / n



def gd(beta_0, X, y, w = None, lr = 1e-2, steps = 10000, plot_loss = False):

    loss_list = [_logistic_loss(X, y, beta_0, w)]

    for _ in range(steps):

        beta_0 = beta_0 - lr * _logistic_loss_grad(X, y, beta_0, w)

        if plot_loss:
            loss_list.append(_logistic_loss(X, y, beta_0, w))
            if len(loss_list) % 500 == 0:
                plt.plot(range( len(loss_list) ), loss_list, label = "training loss")
                plt.legend()
                plt.show()

    return beta_0




# def _logistic_loss_grad(X, y, beta, w = None):
#     if w is None:
#         w = np.ones_like(y)
#
#     margins = y * X @ beta
#
#     return np.mean( w * np.exp( -np.logaddexp(0.0, margins) ) * (-y * X), axis = 0).reshape(-1,1)
#
#
# def _logistic_loss(X, y, beta, w = None):
#     if w is None:
#         w = np.ones_like(y)
#
#     margins = y * X @ beta
#
#     return np.mean( w * np.logaddexp(0.0, -margins) )

#
# def gd(
#     beta_0, X, y, w = None,
#         lr = 1e-2,
#         loss_tol = 1e-4,
#         grad_tol = 1e-4,
#         patience = 10,
#         max_iter = int(1e8)
# ):
#     beta = beta_0.copy()
#
#     # initial loss
#     loss_prev = _logistic_loss(X, y, beta, w)
#     patience_counter = 0
#
#     for t in range(max_iter):
#
#         # Compute new beta and loss
#         beta_new = beta - lr * _logistic_loss_grad(X, y, beta, w)
#         loss_new = _logistic_loss(X, y, beta_new, w)
#
#         # Compute stopping criteria
#         grad = _logistic_loss_grad(X, y, beta_new, w)
#         relative_change = abs(loss_new - loss_prev) / max(1.0, abs(loss_prev))
#         grad_norm = np.linalg.norm(grad)
#
#         # Update parameter and loss
#         beta, loss_prev = beta_new, loss_new
#
#         # Check for stopping
#         if (relative_change < loss_tol) and (grad_norm < grad_tol):
#             patience_counter += 1
#         else:
#             patience_counter = 0
#
#         if patience_counter >= patience:
#             break
#
#         # Warning if hit max iteration
#         if t == max_iter - 1:
#             print(f"Max iteration reached for gradient descent for beta with dimension {beta.shape[0]}.")
#
#     return beta



# def gd(beta_0, X, y, w = None, lr = 1e-2, threshold = 1e-4, plotting = False):
#
#     loss_list = [_logistic_loss(X, y, beta_0, w)]
#     gap = np.inf
#
#     while gap > threshold:   ### Consider a different stopping condition e.g. gradient norm or relative change
#         beta_1 = beta_0 - lr * _logistic_loss_grad(X, y, beta_0, w)
#         gap = np.linalg.norm(beta_1 - beta_0)
#         beta_0 = beta_1
#
#         if plotting:
#
#             loss_list.append(_logistic_loss(X, y, beta_0, w))
#
#             if len(loss_list) % 500 == 0:
#                 num_iter = len(loss_list)
#                 plt.plot(range(num_iter), loss_list, label = "training loss")
#                 plt.legend()
#                 plt.show()
#
#     return beta_0


