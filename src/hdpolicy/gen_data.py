import numpy as np

def _sigmoid(t):
    return (1 + np.e**(-t))**(-1)


def gen_logistic(n, n_dim, rng, beta = None):
    if beta is None:
        beta = rng.normal(0, 1, size = n_dim).reshape(-1, 1)

    X = rng.normal(0, 2, size = (n, n_dim))
    t = X @ beta
    y = ( rng.binomial(1, _sigmoid(t), ) * 2 - 1 ).reshape(-1, 1)

    return X, y, beta


def gen_rct(n, n_dim, rct_prob, rng,  beta_0 = None, beta_1 = None):
    if beta_0 is None:
        beta_0 = rng.normal(-1, 2, size = (n_dim, 1))
    if beta_1 is None:
        beta_1 = rng.normal(1, 2, size = (n_dim, 1))

    X = rng.normal(0, 1, size = (n, n_dim))

    Y0 = X @ beta_0 + rng.normal(0, 1, size=(n, 1))
    Y1 = X @ beta_1 + rng.normal(0, 1, size=(n, 1))

    D = rng.choice([1, -1], size=(n, 1), p=[rct_prob, 1-rct_prob])
    Y = ( (1+D)/2 ) * Y1 + ( (1-D)/2 ) * Y0

    return Y, X, D, Y0, Y1


def gen_nonlinear(n, rct_prob, rng, dgp):

    D = rng.choice([1, -1], size = (n, 1), p = [rct_prob, 1-rct_prob])
    Y0 = rng.normal(0, 1, size = (n, 1))

    if dgp == 'poly_interactions':
        n_main, n_pair, n_triple, n_quad, max_power = 6, 6, 4, 3, 3
        X = rng.normal(0, 1, size = (n, n_main))
        tau = 0

        for j in range(n_main):     # main effect
            p = np.arange(max_power)
            coef = rng.normal(0, 0.3, size = (max_power,1))
            tau += (X[:, j].reshape(-1,1) ** p) @ coef

        all_pairs = [(a, b) for a in range(n_main) for b in range(a+1, n_main)]     # pair interactions
        rng.shuffle(all_pairs)
        powers = []
        for (i, j) in all_pairs[:n_pair]:
            coef = rng.normal(0, 0.3)
            p = rng.integers(1, max_power + 1)
            q = rng.integers(1, max_power + 1)
            tau += coef * (X[:, i]**p).reshape(-1,1) * (X[:, j]**q).reshape(-1,1)
            powers.append( (p,q) )

        all_triplets = [        # triple interactions
            (a, b, c)
            for a in range(n_main)
            for b in range(a + 1, n_main)
            for c in range(b + 1, n_main)
        ]
        rng.shuffle(all_triplets)
        for (i, j, k) in all_triplets[:n_triple]:
            coef = rng.normal(0, 0.3)
            tau += coef * X[:, i].reshape(-1,1) * X[:, j].reshape(-1,1) * X[:, k].reshape(-1,1)

        all_quads = []
        for _ in range(n_quad):     # quad interactions
            idx = rng.choice(n_main, size = 4, replace = False)
            coef = rng.normal(0, 0.2)
            tau += (
                coef * X[:, idx[0]].reshape(-1, 1)
                * X[:, idx[1]].reshape(-1, 1)
                * X[:, idx[2]].reshape(-1, 1)
                * X[:, idx[3]].reshape(-1, 1)
            )
            all_quads.append( idx )

        true_form = (all_pairs, powers, all_triplets, all_quads)

    elif dgp == 'friedman':
        X = rng.uniform(size = (n, 5))
        tau = (
            2.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 4.0 * (X[:, 2] - 0.5)**2
            + 2.0 * X[:, 3] + X[:, 4]
            # + 3.0 * X[:, 0] * X[:, 2] * X[:, 3]
            # + 1.5 * (X[:, 1] > 0.75) * X[:, 4]
        )

        true_form = None

    elif dgp == 'additive_highfreq':
        dim = 6
        X = rng.uniform(size = (n, dim))
        freqs = np.sort( rng.integers(0, 31, size=dim) )
        a = rng.uniform(0.5, 1.5, size=dim)
        b = rng.uniform(0.5, 1.5, size=dim)
        tau = 0
        for j in range(dim):
            tau += (
                a[j] * np.sin(2 * np.pi * freqs[j] * X[:, j])
                + b[j] * np.cos(2 * np.pi * freqs[j] * X[:, j])
            )

        true_form = None

    elif dgp == 'piecewise_const':
        pass

    Y1 = Y0 + tau.reshape(-1,1) + rng.normal(0, 1, size = (n, 1))
    Y = ((1 + D) / 2) * Y1 + ((1 - D) / 2) * Y0

    return Y, X, D, Y0, Y1, tau, true_form


