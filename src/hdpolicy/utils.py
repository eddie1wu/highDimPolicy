import numpy as np

def split_train_test(n_train: int, *arrays):

    train = [arr[:n_train] for arr in arrays]
    test = [arr[n_train:] for arr in arrays]

    return (*train, *test)


def make_weight_and_target(Y, D, rct_prob):

    weight = np.abs(Y) / ( D*rct_prob + (1-D)/2 )
    target = np.sign(Y) * D

    return (weight, target)

