import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from menelaus.detector import StreamingDetector


class LinearFourRates(StreamingDetector):
    """Linear Four Rates detects drift in a learner's true positive rate (TPR),
    true negative rate (TNR), negative predictive value (NPV), and
    positive predictive value (PPV) over time. It relies on the assumption that
    a significant change in any of these rates implies a change in the joint
    distribution of of the features and their classification.

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

    Ref. :cite:t:`wang2015concept`
    """

    input_type = "stream"

    def __init__(
        self,
        time_decay_factor=0.9,
        warning_level=0.05,
        detect_level=0.05,
        burn_in=50,
        num_mc=10000,
        subsample=1,
        rates_tracked=["tpr", "tnr", "ppv", "npv"],
        parallelize=False,
        round_val=4,
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
                decrease the runtime. Defaults to 1.
            rates_tracked (list, optional): A list of the rates that this LFR
                algorithm should track and alert the user based on changes.
                Fewer rates can be tracked based on use case context, as well
                as to improve runtime performance. Defaults to all four rates,
                ["tpr", "tnr", "ppv", "npv"].
            parallelize (boolean, optional): A flag that determines whether
                bound calculations across the rates being tracked by this LFR
                algorithm will be parallelized or not. Advantageous for large
                datasets, but will slow down runtime for fewer data due to
                overhead of threading. Defaults to False.
            round_val: number of decimal points the estimate rate is rounded to
                when stored in bounds dictionary. The greater the ``round_val``, the
                more precise the bounds dictionary will be, and the longer the
                runtime. (Default value = 4)
        """
        super().__init__()
        self.time_decay_factor = time_decay_factor
        self.warning_level = warning_level
        self.detect_level = detect_level
        self.burn_in = burn_in
        self.num_mc = num_mc
        self.subsample = subsample
        self.rates_tracked = rates_tracked
        self.parallelize = parallelize
        self.round_val = round_val
        self.all_drift_states = []
        self._warning_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self._alarm_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self._p_table = {0: {"tpr": 0.5, "tnr": 0.5, "ppv": 0.5, "npv": 0.5}}
        self._bounds = dict()
        self._confusion = np.array([[1, 1], [1, 1]])  # confusion matrix
        self._denominators = {
            0: {"tpr_N": 2, "tnr_N": 2, "ppv_N": 2, "npv_N": 2}
        }  # dictionary of denominators for each statistic at each index
        self._r_stat = (
            self._p_table.copy()
        )  # dictionary of test statistics for P at each index
        self._initialize_retraining_recs()

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self._p_table = {0: {"tpr": 0.5, "tnr": 0.5, "ppv": 0.5, "npv": 0.5}}
        self._confusion = np.array([[1, 1], [1, 1]])  # C at a given time point
        self._denominators = {0: {"tpr_N": 2, "tnr_N": 2, "ppv_N": 2, "npv_N": 2}}
        self._r_stat = self._p_table.copy()
        self._warning_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self._alarm_states = {
            0: {"tpr": False, "tnr": False, "ppv": False, "npv": False}
        }
        self._initialize_retraining_recs()

    # XXX - Order of y_true, y_pred, X differs from abstractmethod signature
    #       for update(). This is done for convenience, so users can call e.g.
    #       LFR.update(1,1) without misinterpretation, but exposes them to a
    #       potential issue where LFR.update(X, y, y) would assign arguments
    #       incorrectly.
    def update(self, y_true, y_pred, X=None):
        """Update detector with a new observation:

        #. Updates confusion matrix (``self._confusion``) with new predictions

        #. Updates the four rates

        #. Test each rate for change over time using bounds from Monte Carlo
           simulations

        #. If any of the rates exceed bounds, change ``drift_state`` to either
           ``"warning"`` or ``"drift"``

        Args:
            y_true: one true label from input data.
            y_pred: one predicted label from input data.
            X: one row of features from input data. Not used in LFR.
        """

        if self.drift_state == "drift":
            self.reset()

        _, y_true, y_pred = super()._validate_input(None, y_true, y_pred)
        super().update(None, y_true, y_pred)
        # the arrays should have a single element after validation.
        y_true, y_pred = y_true[0], y_pred[0]

        y_p = 1 * y_pred
        y_t = 1 * y_true

        old_rates = self._get_four_rates(self._confusion)
        self._confusion[y_p][y_t] += 1
        new_rates = self._get_four_rates(self._confusion)

        # init next index for test stats
        self._r_stat.update(
            {
                self.samples_since_reset: self._r_stat[
                    self.samples_since_reset - 1
                ].copy()
            }
        )
        self._p_table.update(
            {
                self.samples_since_reset: self._p_table[
                    self.samples_since_reset - 1
                ].copy()
            }
        )
        self._warning_states.update(
            {
                self.samples_since_reset: {
                    "tpr": False,
                    "tnr": False,
                    "ppv": False,
                    "npv": False,
                }
            }
        )
        self._alarm_states.update(
            {
                self.samples_since_reset: {
                    "tpr": False,
                    "tnr": False,
                    "ppv": False,
                    "npv": False,
                }
            }
        )

        def _calculate_rate_bounds(rate):
            if new_rates[rate] != old_rates[rate]:
                new_r_stat = self.time_decay_factor * self._r_stat[
                    self.samples_since_reset
                ][rate] + (1 - self.time_decay_factor) * (y_t == y_p)
            else:
                new_r_stat = self._r_stat[self.samples_since_reset - 1][rate]

            self._p_table[self.samples_since_reset][rate] = new_rates[rate]
            self._r_stat[self.samples_since_reset][rate] = new_r_stat
            self._denominators[rate + "_N"] = self._get_four_denominators(
                self._confusion
            )[rate + "_N"]

            if (self.samples_since_reset > self.burn_in) & (
                self.samples_since_reset % self.subsample == 0
            ):
                est_rate = new_rates[rate]
                curr_denom = self._denominators[rate + "_N"]

                r_est_rate = round(est_rate, self.round_val)
                r_curr_denom = round(curr_denom, self.round_val)

                bound_dict = self._update_bounds_dict(
                    est_rate, curr_denom, r_est_rate, r_curr_denom
                )

                lb_warn = bound_dict["lb_warn"]
                ub_warn = bound_dict["ub_warn"]
                lb_detect = bound_dict["lb_detect"]
                ub_detect = bound_dict["ub_detect"]

                self._warning_states[self.samples_since_reset][rate] = (
                    new_r_stat < lb_warn
                ) | (new_r_stat > ub_warn)
                self._alarm_states[self.samples_since_reset][rate] = (
                    new_r_stat < lb_detect
                ) | (new_r_stat > ub_detect)

        if self.parallelize:
            Parallel(n_jobs=2, require="sharedmem")(
                delayed(_calculate_rate_bounds)(rate) for rate in self.rates_tracked
            )
        else:
            for rate in self.rates_tracked:
                _calculate_rate_bounds(rate)

        if any(self._alarm_states[self.samples_since_reset].values()):
            self.all_drift_states.append("drift")
            self.drift_state = "drift"
        elif any(self._warning_states[self.samples_since_reset].values()):
            self.all_drift_states.append("warning")
            self.drift_state = "warning"
        else:
            self.all_drift_states.append(None)
            self.drift_state = None

        if self.drift_state is not None:
            self._increment_retraining_recs()

    def _initialize_retraining_recs(self):
        """Sets ``self._retraining_recs`` to ``[None, None]``."""
        self._retraining_recs = [None, None]

    def _increment_retraining_recs(self):
        """Set ``self._retraining_recs`` to the beginning and end of the current
        drift/warning region.
        """
        if self.drift_state == "warning" and self._retraining_recs[0] is None:
            self._retraining_recs[0] = self.total_samples - 1

        if self.drift_state == "drift":
            self._retraining_recs[1] = self.total_samples - 1
            if self._retraining_recs[0] is None:
                self._retraining_recs[0] = self.total_samples - 1

    @staticmethod
    def _get_four_rates(confusion):
        """Takes a confusion matrix and returns a dictionary with values
        for TPR, TNR, PPV, NPV.

        Args:
            confusion: matrix with TN located at [0,0], FN at [0,1], FP at
                [1,0], and TP at [1,1]

        Returns:
            dict: a dictionary with TPR, TNR, PPV, NPV.
        """
        tn, fn, fp, tp = confusion.ravel()
        result = dict()
        result["tpr"] = tp / (tp + fn)
        result["tnr"] = tn / (tn + fp)
        result["ppv"] = tp / (fp + tp)
        result["npv"] = tn / (tn + fn)
        return result

    @staticmethod
    def _get_four_denominators(confusion):
        """
        Takes a confusion matrix and returns a dictionary with denominators
        for TPR, TNR, PPV, NPV.

        Args:
            confusion: matrix with TN located at [0,0], FN at [0,1], FP at [1,0],
            and TP at [1,1]

        Returns:
            dict: a dictionary with denominators for TPR, TNR, PPV, NPV.
        """
        tn, fn, fp, tp = confusion.ravel()
        result = dict()
        result["tpr_N"] = tp + fn
        result["tnr_N"] = tn + fp
        result["ppv_N"] = fp + tp
        result["npv_N"] = tn + fn
        return result

    def _update_bounds_dict(self, est_rate, curr_denom, r_est_rate, r_curr_denom):
        """
        Checks if combination of rounded ``est_rate`` and denom has been seen before.
        If yes, reuse the bounds estimates. If no, simulate new bounds estimates
        and maintain in sorted bound dictionary. This method calculates Monte
        Carlo simulations using exact rates but stores results using rounded
        value.

        Args:
            est_rate: empirical estimate of rate (P)
            curr_denom: denominator of rate
            r_est_rate: rounded ``est_rate``
            r_curr_denom: rounded denom

        Returns:
            dict: dictionary storing the bounds from MonteCarlo simulation for
                each rate for previously seen pairs of estimated empirical rate
                and denom (time steps) with the structure:
                {rate: {N_1: bound1, N_2: bound2, ...}}
        """
        if r_est_rate in self._bounds:
            denom_dict = self._bounds[r_est_rate]

            if r_curr_denom in denom_dict:
                bound_dict = denom_dict[r_curr_denom]
            else:
                bound_dict = self._sim_bounds(est_rate, curr_denom)
                denom_dict[r_curr_denom] = bound_dict
                self._bounds[r_est_rate] = denom_dict
                denom_dict = dict(sorted(denom_dict.items()))
        else:
            bound_dict = self._sim_bounds(est_rate, curr_denom)

            denom_dict = {r_curr_denom: bound_dict}
            denom_dict[r_curr_denom] = bound_dict
            self._bounds[r_est_rate] = denom_dict
            self._bounds = dict(sorted(self._bounds.items()))

        return bound_dict

    def _sim_bounds(self, est_rate, denom):
        """
        Takes an estimated rate and number of time steps denom and returns
        dictionary of lower and upper bounds for its empirical distribution.

        Args:
            est_rate: empirical estimate of rate (P)
            denom: denominator of rate

        Returns:
            dict: dictionary with keys ``['lb_warn', 'ub_warn', 'lb_detect',
                'ub_detect']`` corresponding to the lower and upper bounds at
                the respective thresholds
        """
        eta = self.time_decay_factor
        warning_level = self.warning_level
        detect_level = self.detect_level
        num_mc = self.num_mc

        exps = [denom - i for i in range(1, denom + 1)]
        prods = [
            eta ** (exps[i]) for i in range(denom)
        ]  # eta^(denom - i) where i is from 1 to denom
        result_matrix = pd.DataFrame(
            np.repeat(prods, num_mc).reshape(len(prods), num_mc)
        )

        def get_Rj(vec, eta, est_rate, denom):
            """Get modified rate as a test statistic for the empirical rate

            Args:
              vec: vector to re-weight by result of bernoulli trials
              eta: time decay factor (see ``self.time_decay_factor``)
              est_rate: current estimated rate
              denom: current denominator for the rate

            Returns:

            """
            # est_rate and size seem like they could be inferred from the passed
            # vector and self.time_decay_rate
            bools = np.random.binomial(n=1, p=est_rate, size=denom)
            return (1 - eta) * sum(vec * bools)

        result_vector = result_matrix.apply(get_Rj, axis=0, args=(eta, est_rate, denom))

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

    @property
    def retraining_recs(self):
        """Recommended indices between the first warning and drift for
        retraining. Resets during return to normal state after each detection
        of drift. If no warning fires, recommendation is from current drift
        -> current drift. In this case, caution is urged when retraining, as
        this situation indicates an abrupt change.

        Returns:
            list: the current retraining recommendations
        """
        return self._retraining_recs
