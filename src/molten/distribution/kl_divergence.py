import numpy as np


def kl_divergence(vals_p, vals_q, discrete=True):
    """Computes Kullback-Leibler divergence between two distributions

    Args:
        vals_p (list): List of values from first distribution
        vals_q (list): List of values from second distribution
        discrete (bool, optional): If True, assumes nonparametric estimation.
         If False, assumes normal distribution for both sets. Defaults to True.

    Returns:
        float: Kullback-Leibler divergence
    """

    if discrete:
        dist = sum([x * np.log(x / y) if x * y != 0 else 0 for x, y in zip(vals_p, vals_q)])

    else:

        p_mean = np.mean(vals_p)
        q_mean = np.mean(vals_q)

        p_var = np.var(vals_p)
        q_var = np.var(vals_q)

        dist = (((p_mean - q_mean) ** 2) / (2 * p_var)) + (1 / 2) * (
            (q_var / p_var) - 1 - np.log(q_var / p_var)
        )

    return dist
