import numpy as np


def kl_divergence(p, q, type="discrete"):
    """Computes Kullback-Leibler divergence between two distributions

    Args:
        p (list): List of values from first distribution
        q (list): List of values from second distribution
        type (str, optional): String of either "discrete" or "continuous",
            describing the type of distribution. Defaults to "discrete".

    Returns:
        float: Kullback-Leibler divergence
    """

    if type == "discrete":
        dist = sum([x * np.log(x / y) for x, y in zip(p, q)])

    if type == "continuous":

        p_mean = np.mean(p)
        q_mean = np.mean(q)

        p_var = np.var(p)
        q_var = np.var(q)

        dist = (((p_mean - q_mean) ** 2) / (2 * p_var)) + (1 / 2) * (
            (q_var / p_var) - 1 - np.log(q_var / p_var)
        )

    return dist
