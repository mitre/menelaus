import numpy as np
from scipy.stats import ks_2samp


STATE_INIT = "baseline"
STATE_DRIFT = "alarm"
STATE_WARNING = "warning"


class KolmogorovSmirnovAlarm:
    def __init__(self, alpha=0.05, critical_feature_proportion=0.25):
        self._state = STATE_INIT
        self.alpha = alpha
        self.critical_feature_proportion = critical_feature_proportion

    def evaluate(self, rep_reference, rep_test):
        rep_reference = rep_reference.reshape(rep_reference.shape[0], -1)
        rep_test = rep_test.reshape(rep_test.shape[0], -1)
        if rep_reference.shape[0] != rep_test.shape[0]:
            raise ValueError(f"...")
        n_features = rep_reference.shape[1]
        p_values = np.zeros(n_features, dtype=np.float32)
        # distances = np.zeros_like(p_values)
        for i in range(n_features):
            _, p_values[i] = ks_2samp(
                rep_reference[:, i],
                rep_test[:, i],
                alternative="two-sided",
                mode="asymp",
            )
            p_values[i] /= n_features
        extreme_features_proportion = (
            len([p for p in p_values if p <= self.alpha]) / n_features
        )
        if 0 < extreme_features_proportion < self.critical_feature_proportion:
            self._state = STATE_WARNING
        elif extreme_features_proportion >= self.critical_feature_proportion:
            self._state = STATE_DRIFT
        else:
            self._state = STATE_INIT
        print(p_values)
