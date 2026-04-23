import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Stable sigmoid
# -----------------------------
def sigmoid(z):
    out = np.empty_like(z)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)

    return out


# -----------------------------
# Data generating process
# P(Y=1|x) = sigmoid(x + x^2)
# labels in {-1, +1}
# -----------------------------
def gen_logistic(n, rng):
    X = rng.normal(0, 1, size=(n, 1))

    t = X + X**2
    p = sigmoid(t)

    y = rng.binomial(1, p)
    y = 2 * y - 1

    return X, y.reshape(-1, 1).astype(float)


# -----------------------------
# Piecewise linear spline features
# Basis:
#   1, x, (x-t1)_+, ..., (x-tm)_+
# where (u)_+ = max(u, 0)
# -----------------------------
def spline_features(X, knots):
    # X: (n,1)
    # knots: (m,)
    hinge = np.maximum(X - knots.reshape(1, -1), 0.0)
    Phi = np.hstack([np.ones((X.shape[0], 1)), X, hinge])
    return Phi


# -----------------------------
# Standardize non-bias columns
# Do not standardize the intercept
# -----------------------------
def standardize_train_test(Phi_train, Phi_test):
    Phi_train_std = Phi_train.copy()
    Phi_test_std = Phi_test.copy()

    mean = Phi_train_std[:, 1:].mean(axis=0, keepdims=True)
    std = Phi_train_std[:, 1:].std(axis=0, keepdims=True) + 1e-8

    Phi_train_std[:, 1:] = (Phi_train_std[:, 1:] - mean) / std
    Phi_test_std[:, 1:] = (Phi_test_std[:, 1:] - mean) / std

    return Phi_train_std, Phi_test_std


# -----------------------------
# Logistic loss
# -----------------------------
def logistic_loss(Phi, y, w):
    margin = y * (Phi @ w)
    return np.mean(np.log1p(np.exp(-margin)))


# -----------------------------
# Logistic GD for linear classifier
# -----------------------------
def train_linear_gd(Phi, y, lr=0.05, steps=50000):
    n, p = Phi.shape
    w = np.zeros((p, 1))

    for _ in range(steps):
        margin = y * (Phi @ w)
        grad = -(Phi.T @ (y * sigmoid(-margin))) / n
        w -= lr * grad

    return w


# -----------------------------
# Classification risk
# -----------------------------
def classification_risk(y_true, y_pred):
    return np.mean(y_true != y_pred)


# -----------------------------
# Prediction
# -----------------------------
def predict(Phi, w):
    scores = Phi @ w
    return np.where(scores >= 0, 1, -1).reshape(-1, 1)


# -----------------------------
# Experiment
# -----------------------------
def run_experiment():
    rng = np.random.default_rng(0)

    n_train = 100
    n_test = 10000

    X_train, y_train = gen_logistic(n_train, rng)
    X_test, y_test = gen_logistic(n_test, rng)

    # number of spline knots
    knot_counts = np.arange(0, 2000, 100)

    train_risks = []
    test_risks = []
    train_losses = []
    n_features = []

    # use a fixed large knot dictionary so the sieve is nested
    # choose knots over a broad interval covering most of the Gaussian mass
    max_knots = int(knot_counts.max())
    knot_grid_full = np.linspace(-3.5, 3.5, max_knots) if max_knots > 0 else np.array([])

    for m in tqdm(knot_counts):
        knots = knot_grid_full[:m]

        Phi_train = spline_features(X_train, knots)
        Phi_test = spline_features(X_test, knots)

        Phi_train, Phi_test = standardize_train_test(Phi_train, Phi_test)

        n_features.append(Phi_train.shape[1])

        w = train_linear_gd(Phi_train, y_train, lr=0.05, steps=50000)

        train_pred = predict(Phi_train, w)
        test_pred = predict(Phi_test, w)

        train_risks.append(classification_risk(y_train, train_pred))
        test_risks.append(classification_risk(y_test, test_pred))
        train_losses.append(logistic_loss(Phi_train, y_train, w))

    return n_features, train_risks, test_risks, train_losses, n_train


# -----------------------------
# Run experiment
# -----------------------------
p, train_risk, test_risk, train_loss, n_train = run_experiment()


# -----------------------------
# Plot classification risk
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(p, train_risk, marker="o", label="Training risk")
plt.plot(p, test_risk, marker="o", label="Test risk")
plt.axvline(n_train, linestyle="--", label="n_train")

plt.xlabel("Number of spline features")
plt.ylabel("Classification risk")
plt.title("Double Descent with Linear Spline Sieve")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Plot training logistic loss
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(p, train_loss, marker="o", label="Training logistic loss")
plt.axvline(n_train, linestyle="--", label="n_train")

plt.xlabel("Number of spline features")
plt.ylabel("Logistic loss")
plt.title("Training Logistic Loss with Linear Spline Sieve")
plt.legend()
plt.grid(True)
plt.show()
