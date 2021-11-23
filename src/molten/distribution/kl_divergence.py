import numpy as np


def kl_divergence(vals_p, vals_q, d_type="discrete"):
    """Computes Kullback-Leibler divergence between two distributions

    Args:
        vals_p (list): List of values from first distribution
        vals_q (list): List of values from second distribution
        d_type (str, optional): String of either "discrete" or "continuous",
            describing the type of distribution. Defaults to "discrete".

    Returns:
        float: Kullback-Leibler divergence
    """

    if d_type == "discrete":
        dist = sum([x * np.log(x / y) for x, y in zip(vals_p, vals_q)])

    if d_type == "continuous":

        p_mean = np.mean(vals_p)
        q_mean = np.mean(vals_q)

        p_var = np.var(vals_p)
        q_var = np.var(vals_q)

        dist = (((p_mean - q_mean) ** 2) / (2 * p_var)) + (1 / 2) * (
            (q_var / p_var) - 1 - np.log(q_var / p_var)
        )

    return dist
