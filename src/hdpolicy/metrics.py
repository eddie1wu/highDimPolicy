import numpy as np

def compute_welfare(tau, Y0, Y1):

    policy = np.sign(tau)
    welfare = np.mean( (1+policy)/2 * Y1 + (1-policy)/2 * Y0 )

    return welfare

