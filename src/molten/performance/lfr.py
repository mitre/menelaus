import pandas as pd
import numpy as np
from molten.DriftDetector import DriftDetector


class LinearFourRates(DriftDetector):
    """Linear Four Rates detects drift in a learner's true positive rate, true
    negative rate, negative predictive value, and positive predictive value over
    time. It relies on the assumption that a significant change in any of these
    rates implies a change in the joint distribution of of the features and
    their classification.

    For each rate, the empirical rate is calculated at each sample. The test
    statistic for each rate is a weighted average of all observed empirical
    rates, which is used to test the hypothesis that the distribution of the
    given rate at time t-1 is equal to the distribution of the rate at time t.
    More accurate estimates for the bounds of these empirical rates are obtained
    by Monte Carlo simulation.

    This implementation incorporates a semi-offline bounds dictionary to reduce
    runtime. Instead of running the Monte Carlo simulations for each combination
    of number of time steps and estimated empirical rate, if a given combination
    has been simulated before, the bounds are re-used.

    Ref. W. Heng and Z. Abraham, “Concept drift detection for streaming
    data,” in Proc. 2015 Int. Joint Conf. Neural Networks, 2015, Conference
    Proceedings, pp. 1–9.

    Attributes:
        warning_states: dictionary of warning state for each rate at each index
        alarm_states: dictionary of alarm state for each rate at each index
        P: dictionary of values for each of the four rates at each index
        bounds: dictionary storing the bounds from MonteCarlo simulation for
            each rate for previously seen pairs of estimated empirical rate and
            N (time steps) with the structure:
                {rate: {N_1: bound1, N_2: bound2, ...}}
        retraining_recs: recommends indexes between first warning and drift for
            retraining. Resets during return to normal state after each detection
            of drift. If no warning alarms, recommendation is from current drift
            -> current drift. Urge caution when retraining as this situation
            indicates an abrupt change.

    """

    def __init__(
        self,
        time_decay_factor=0.9,
        warning_level=0.05,
        detect_level=0.05,
        burn_in=50,
        num_mc=10000,
        subsample=None,
    ):
        """
        Args:
            time_decay_factor (float, optional): Amount of weight given to
                current timepoint, must be in [0,1]. The smaller the value, the
                more conservative in identifying drift and the less weight given
                to abrupt changes. Defaults to 0.9.
            warning_level (float, optional): Statistical significance level for
                warnings. Defaults to 0.05.
            detect_level (float, optional): Statistical significance level for
                detection. Defaults to 0.05.
            burn_in (int, optional): Number of observations to make up a burn-in
                period; simulations will not happen until this index has passed,
                initially and after reaching drift state. Defaults to 50.
                num_mc (int, optional): Number of Monte Carlo iterations to run.
                Defaults to 10000.
            subsample (int, optional): A subsample of value n will only test for
                drift every nth observation. Rates will still be calculated, the
                monte carlo simulation will not be. Larger subsample value will
                decrease the runtime. Defaults to None.
        """
        super().__init__()
        self.time_decay_factor = time_decay_factor
        self.warning_level = warning_level
        self.detect_level = detect_level
        self.num_mc = num_mc
        self.burn_in = burn_in
        if self.burn_in is None:
            self.burn_in = 0
        self.subsample = subsample
        self.all_drift_states = []
        self.warning_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self.alarm_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self.P = {0: {"tpr": 0.5, "tnr": 0.5, "ppv": 0.5, "npv": 0.5}}
        self.bounds = dict()
        self._C = np.array([[1, 1], [1, 1]])  # confusion matrix
        self._N = {
            0: {"tpr_N": 2, "tnr_N": 2, "ppv_N": 2, "npv_N": 2}
        }  # dictionary of denominators for each statistic at each index
        self._R = self.P.copy()  # dictionary of test statistics for P at each index
        self._initialize_retraining_recs()

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after drift_state == 'drift'.
        """
        super().reset()
        self.P = {0: {"tpr": 0.5, "tnr": 0.5, "ppv": 0.5, "npv": 0.5}}
        self._C = np.array([[1, 1], [1, 1]])  # C at a given time point
        self._N = {0: {"tpr_N": 2, "tnr_N": 2, "ppv_N": 2, "npv_N": 2}}
        self._R = self.P.copy()
        self.warning_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self.alarm_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self._initialize_retraining_recs()

    def update(self, y_pred, y_true, round_val=4):
        """Update detector with a new observation:
            1. Updates confusion matrix (self._C) with new predictions
            2. Updates the four rates
            3. Test each rate for change over time using bounds from Monte Carlo
                simulations
            4. If any of the rates exceed bounds, change drift_state to either
                "warning" or "drift"

        Args:
            y_pred: predicted class
            y_true: actual class
            round_val: number of decimal points the estimate rate is rounded to
                when stored in bounds dictionary. The greater the round_val, the
                more precise the bounds dictionary will be, and the longer the
                runtime. (Default value = 4)
        """
        if self.drift_state == "drift":
            self.reset()

        super().update()
        y_t = 1 * y_true
        yhat_t = 1 * y_pred

        old_C = self._C.copy()
        self._C[yhat_t][y_t] += 1

        old_rates = self.get_four_rates(old_C)
        new_rates = self.get_four_rates(self._C)

        # init next index for test stats
        self._R.update({self.n: self._R[self.n - 1].copy()})
        self.P.update({self.n: self.P[self.n - 1].copy()})
        self.warning_states.update(
            {self.n: {"tpr": False, "tnr": False, "ppv": False, "npv": False}}
        )
        self.alarm_states.update(
            {self.n: {"tpr": False, "tnr": False, "ppv": False, "npv": False}}
        )

        for rate in ["tpr", "tnr", "ppv", "npv"]:
            if new_rates[rate] != old_rates[rate]:
                new_R = self.time_decay_factor * self._R[self.n][rate] + (
                    1 - self.time_decay_factor
                ) * (y_t == yhat_t)
            else:
                new_R = self._R[self.n - 1][rate]

            self.P[self.n][rate] = new_rates[rate]
            self._R[self.n][rate] = new_R
            self._N[rate + "_N"] = self.get_four_denominators(self._C)[rate + "_N"]

            if (self.n > self.burn_in) & (self.n % self.subsample == 0):
                est_rate = new_rates[rate]
                N = self._N[rate + "_N"]

                r_est_rate = round(est_rate, round_val)
                r_N = round(N, round_val)

                bound_dict = self.update_bounds_dict(est_rate, N, r_est_rate, r_N)

                lb_warn = bound_dict["lb_warn"]
                ub_warn = bound_dict["ub_warn"]
                lb_detect = bound_dict["lb_detect"]
                ub_detect = bound_dict["ub_detect"]

                self.warning_states[self.n][rate] = (new_R < lb_warn) | (
                    new_R > ub_warn
                )
                self.alarm_states[self.n][rate] = (new_R < lb_detect) | (
                    new_R > ub_detect
                )

        if any(self.alarm_states[self.n].values()):
            self.all_drift_states.append("drift")
            self.drift_state = "drift"
        elif any(self.warning_states[self.n].values()):
            self.all_drift_states.append("warning")
            self.drift_state = "warning"
        else:
            self.all_drift_states.append(None)
            self.drift_state = None

        if self.drift_state is not None:
            self._increment_retraining_recs()

    def _initialize_retraining_recs(self):
        """ """
        self.retraining_recs = [None, None]

    def _increment_retraining_recs(self):
        """Default retraining recommendation is [warning index, drift index]. If
        no warning occurs, this will instead be [drift index, drift index]. Be
        cautious, as this indicates an abrupt change.
        """
        if self.drift_state == "warning" and self.retraining_recs[0] is None:
            self.retraining_recs[0] = self.total_samples - 1

        if self.drift_state == "drift":
            self.retraining_recs[1] = self.total_samples - 1
            if self.retraining_recs[0] is None:
                self.retraining_recs[0] = self.total_samples - 1

    def get_four_rates(self, confusion):
        """Takes a confusion matrix and returns a dictionary with TPR, TNR, PPV,
        NPV.

        Args:
            confusion: matrix with TN located at [0,0], FN at [0,1], FP at
                [1,0], and TP at [1,1]

        Returns:
            dict: a dictionary with TPR, TNR, PPV, NPV.
        """
        tn, fn, fp, tp = confusion.ravel()
        d = dict()
        d["tpr"] = tp / (tp + fn)
        d["tnr"] = tn / (tn + fp)
        d["ppv"] = tp / (fp + tp)
        d["npv"] = tn / (tn + fn)
        return d

    def get_four_denominators(self, confusion):
        """Takes a confusion matrix and returns a dictionary with denominators
        for TPR, TNR, PPV, NPV.

        Args:
            confusion: matrix with TN located at [0,0], FN at [0,1], FP at [1,0],
            and TP at [1,1]

        Returns:
            dict: a dictionary with denominators for TPR, TNR, PPV, NPV.
        """
        tn, fn, fp, tp = confusion.ravel()
        d = dict()
        d["tpr_N"] = tp + fn
        d["tnr_N"] = tn + fp
        d["ppv_N"] = fp + tp
        d["npv_N"] = tn + fn
        return d

    def update_bounds_dict(self, est_rate, N, r_est_rate, r_N):
        """Checks if combination of rounded est_rate and N has been seen before.
        If yes, reuse the bounds estimates. If no, simulate new bounds estimates
        and maintain in sorted bound dictionary. This method calculates Monte
        Carlo simulations using exact rates but stores results using rounded
        value.

        Args:
            est_rate: empirical estimate of rate (P)
            N: denominator of rate
            r_est_rate: rounded est_rate
            r_N: rounded N

        Returns:
            dict: dictionary storing the bounds from MonteCarlo simulation for
                each rate for previously seen pairs of estimated empirical rate
                and N (time steps) with the structure:
                    {rate: {N_1: bound1, N_2: bound2, ...}}
        """
        if r_est_rate in self.bounds:
            N_dict = self.bounds[r_est_rate]

            if r_N in N_dict:
                bound_dict = N_dict[r_N]
            else:
                bound_dict = self.sim_bounds(est_rate, N)
                N_dict[r_N] = bound_dict
                N_dict = dict(sorted(N_dict.items()))
                self.bounds[r_est_rate] = N_dict
        else:
            bound_dict = self.sim_bounds(est_rate, N)

            N_dict = {r_N: bound_dict}
            self.bounds[r_est_rate] = N_dict
            self.bounds = dict(sorted(self.bounds.items()))

        return bound_dict

    def sim_bounds(self, est_rate, N):
        """Takes an estimated rate and number of time steps N and returns
        dictionary of lower and upper bounds for its empirical distribution.

        Args:
            est_rate: empirical estimate of rate (P)
            N: denominator of rate

        Returns:
            dict: dictionary with keys ['lb_warn', 'ub_warn', 'lb_detect',
                'ub_detect'] correpsonding to the lower and upper bounds at
                the respective thresholds


        """
        eta = self.time_decay_factor
        warning_level = self.warning_level
        detect_level = self.detect_level
        num_mc = self.num_mc

        exps = [N - i for i in range(1, N + 1)]
        prods = [
            eta ** (exps[i]) for i in range(N)
        ]  # eta^(N - i) where i is from 1 to N
        result_matrix = pd.DataFrame(
            np.repeat(prods, num_mc).reshape(len(prods), num_mc)
        )

        def get_Rj(vec, eta, est_rate, N):
            """

            Args:
              vec:
              eta:
              est_rate:
              N:

            Returns:

            """
            bools = np.random.binomial(n=1, p=est_rate, size=N)
            return (1 - eta) * sum(vec * bools)

        result_vector = result_matrix.apply(get_Rj, axis=0, args=(eta, est_rate, N))

        # find lower and upper bound for each alpha level
        lb_warn = np.percentile(result_vector, q=warning_level * 100)
        ub_warn = np.percentile(result_vector, q=100 - (warning_level * 100))
        lb_detect = np.percentile(result_vector, q=detect_level * 100)
        ub_detect = np.percentile(result_vector, q=100 - (detect_level * 100))

        bounds = {
            "lb_warn": lb_warn,
            "ub_warn": ub_warn,
            "lb_detect": lb_detect,
            "ub_detect": ub_detect,
        }
        return bounds
