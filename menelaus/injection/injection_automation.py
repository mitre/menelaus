import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import sklearn
from scipy.io.arff import loadarff

from menelaus.concept_drift import LinearFourRates, ADWINAccuracy, DDM, EDDM, STEPD, MD3
from menelaus.data_drift import PCACD, KdqTreeStreaming, KdqTreeBatch, NNDVI
from menelaus.data_drift.cdbd import CDBD
from menelaus.data_drift.hdddm import HDDDM
import label_manipulation
import feature_manipulation
import noise


def select_random_classes(series):
    classes = series.unique()

    if len(classes) < 2:
        raise ValueError(f"Insufficient classes in series: {len(classes)}")
    else:
        class_a = classes[random.randint(0, len(classes) - 1)]
        class_b = classes[random.randint(0, len(classes) - 1)]

        while class_a == class_b:
            class_b = classes[random.randint(0, len(classes) - 1)]

        return [class_a, class_b]


class InjectionTesting:
    def __init__(self, data_path, seed=None, numeric_cols=None, categorical_cols=None):
        file_type = data_path.split(".")[-1]
        self.seed = seed
        self.numeric_cols = []
        self.categorical_cols = []

        if file_type == "csv":
            self.df = pd.read_csv(data_path)
        elif file_type == "arff":
            raw_data = loadarff(data_path)
            self.df = pd.DataFrame(raw_data[0])
        else:
            raise ValueError(f"Invalid file type: {file_type}")

        if not numeric_cols or not categorical_cols:
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]) and numeric_cols is None:
                    self.numeric_cols.append(col)
                elif self.df[col].nunique() < len(self.df) and categorical_cols is None:
                    self.categorical_cols.append(col)
        if numeric_cols:
            self.numeric_cols = numeric_cols
        if categorical_cols:
            self.categorical_cols = categorical_cols

        if seed:
            random.seed(seed)

    def select_rows(self, start, end):
        start_row = int(start * len(self.df))
        end_row = int(end * len(self.df))

        return [start_row, end_row]

    def train_linear_model(self, x_cols=None, y_col=None, start=0, end=0.75):
        if not x_cols or not y_col:
            y_col = self.numeric_cols[random.randint(0, len(self.numeric_cols) - 1)]
            x_cols = self.numeric_cols.copy()
            x_cols.remove(y_col)

        model = sklearn.linear_model.LinearRegression()
        start_train, end_train = self.select_rows(start, end)
        train_df = self.df.iloc[
            start_train:end_train,
        ]
        model.fit(train_df[x_cols], train_df[y_col])

        return model, x_cols, y_col

    def train_classifier_model(
        self,
        model_type="svc",
        x_cols=None,
        y_col=None,
        start=0,
        end=0.75,
        limit_classes=None,
    ):
        if not x_cols or not y_col:
            y_col = self.categorical_cols[
                random.randint(0, len(self.categorical_cols) - 1)
            ]
            x_cols = self.numeric_cols.copy()

        encoder = sklearn.preprocessing.LabelEncoder()
        encoder.fit(self.df[y_col])
        self.df[f"{y_col}_encoded"] = encoder.transform(self.df[y_col])
        y_col = f"{y_col}_encoded"

        if limit_classes:
            self.df = self.df[self.df[y_col] < limit_classes]

        if model_type == "svc":
            model = sklearn.svm.SVC(kernel="linear")
        elif model_type == "logistic":
            model = sklearn.linear_model.LogisticRegression()
        else:
            raise ValueError(f"Model type not supported: {model_type}")

        start_train, end_train = self.select_rows(start, end)
        train_df = self.df.iloc[
            start_train:end_train,
        ]
        model.fit(train_df[x_cols], train_df[y_col])

        return model, x_cols, y_col

    def inject_random_brownian_noise(self, x, start=0.75, end=1, num_drift_cols=1):
        rand_cols = []
        start_drift, end_drift = self.select_rows(start, end)

        for i in range(num_drift_cols):
            rand_col = self.numeric_cols[random.randint(0, len(self.numeric_cols) - 1)]
            rand_cols.append(rand_col)

            self.df = noise.BrownianNoiseInjector.__call__(self.df, rand_col, x, start_drift, end_drift)

        return rand_cols

    def inject_random_class_manipulation(
        self, manipulation_type, start=0.75, end=1, num_drift_cols=1
    ):
        rand_cols = []
        all_rand_classes = []
        start_drift, end_drift = self.select_rows(start, end)

        for i in range(num_drift_cols):
            rand_col = self.categorical_cols[
                random.randint(0, len(self.categorical_cols) - 1)
            ]
            rand_cols.append(rand_col)
            rand_classes = select_random_classes(self.df[rand_col])
            all_rand_classes.append(rand_classes)

            if manipulation_type == "class_swap":
                self.df = label_manipulation.LabelSwapInjector().__call__(
                    self.df,
                    rand_col,
                    rand_classes[0],
                    rand_classes[1],
                    start_drift,
                    end_drift,
                )
            elif manipulation_type == "class_join":
                new_label = f"{rand_classes[0]}_{rand_classes[1]}"
                self.df = label_manipulation.LabelJoinInjector().__call__(
                    self.df,
                    rand_col,
                    rand_classes[0],
                    rand_classes[1],
                    new_label,
                    start_drift,
                    end_drift,
                )
            else:
                raise ValueError(
                    f"Invalid class manipulation type: {manipulation_type}"
                )

        return rand_cols, all_rand_classes

    def inject_random_feature_swap(self, start=0.75, end=1, num_swaps=1):
        all_swap_cols = []
        start_drift, end_drift = self.select_rows(start, end)

        for i in range(num_swaps):
            col_type = (
                self.numeric_cols
                if random.randint(0, 1) == 0
                else self.categorical_cols
            )

            if len(col_type) < 2:
                col_type = (
                    self.numeric_cols
                    if col_type == self.categorical_cols
                    else self.categorical_cols
                )
            if len(col_type) < 2:
                raise ValueError(
                    "Insufficient numeric and categorical columns for swaps"
                )

            col_a = col_type[random.randint(0, len(col_type) - 1)]
            col_b = col_type[random.randint(0, len(col_type) - 1)]

            while col_a == col_b:
                col_b = col_type[random.randint(0, len(col_type) - 1)]

            swap_cols = [col_a, col_b]
            all_swap_cols.append(swap_cols)
            self.df = feature_manipulation.FeatureSwapInjector().__call__(
                self.df, col_a, col_b, start_drift, end_drift
            )

        return all_swap_cols

    def inject_random_feature_hide_and_sample(self):
        rand_col = self.df.columns[random.randint(0, len(self.df.columns) - 1)]
        sample_size = min(self.df[rand_col].value_counts())
        self.df = feature_manipulation.FeatureCoverInjector().__call__(
            self.df, rand_col, sample_size
        )

        return rand_col

    def test_adwin_detector(self, model=None, x_cols=None, y_col=None):
        if not model:
            model, x_cols, y_col = self.train_linear_model(x_cols=x_cols, y_col=y_col)

        self.df["y_pred"] = model.predict(self.df[x_cols])
        detector = ADWINAccuracy()
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(X=row[x_cols], y_true=row[y_col], y_pred=row["y_pred"])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def test_cbdb_detector(self, cols, group_col=None, reference_group=None, subsets=8):
        if not group_col:
            group_col = self.categorical_cols[
                random.randint(0, len(self.categorical_cols) - 1)
            ]

            while group_col in cols:
                group_col = self.categorical_cols[
                    random.randint(0, len(self.categorical_cols) - 1)
                ]

        if not reference_group:
            reference_group = self.df[group_col].min()

        reference_df = self.df[self.df[group_col] == reference_group][cols]
        test_df = self.df[self.df[group_col] != reference_group]
        detector = CDBD(subsets=subsets)
        detector.set_reference(reference_df)
        drift_state = []

        for group_id, subset_data in test_df.groupby(group_col):
            detector.update(subset_data[cols])
            drift_state.append(detector.drift_state)

        return detector, drift_state

    def test_ddm_detector(
        self,
        model=None,
        x_cols=None,
        y_col=None,
        n_threshold=100,
        warning_scale=7,
        drift_scale=10,
    ):
        if not model:
            model, x_cols, y_col = self.train_classifier_model(
                model_type="svc", x_cols=x_cols, y_col=y_col
            )

        self.df["y_pred"] = model.predict(self.df[x_cols])
        detector = DDM(
            n_threshold=n_threshold,
            warning_scale=warning_scale,
            drift_scale=drift_scale,
        )
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(y_true=row[y_col], y_pred=row["y_pred"])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def test_eddm_detector(
        self,
        model=None,
        x_cols=None,
        y_col=None,
        n_threshold=30,
        warning_thresh=0.7,
        drift_thresh=0.5,
    ):
        if not model:
            model, x_cols, y_col = self.train_classifier_model(
                model_type="svc", x_cols=x_cols, y_col=y_col
            )

        self.df["y_pred"] = model.predict(self.df[x_cols])
        detector = EDDM(
            n_threshold=n_threshold,
            warning_thresh=warning_thresh,
            drift_thresh=drift_thresh,
        )
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(y_true=row[y_col], y_pred=row["y_pred"])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def test_hdddm_detector(
        self, cols, group_col=None, reference_group=None, subsets=8
    ):
        if not group_col:
            group_col = self.categorical_cols[
                random.randint(0, len(self.categorical_cols) - 1)
            ]

            while group_col in cols:
                group_col = self.categorical_cols[
                    random.randint(0, len(self.categorical_cols) - 1)
                ]

        if not reference_group:
            reference_group = self.df[group_col].min()

        reference_df = self.df[self.df[group_col] == reference_group][cols]
        test_df = self.df[self.df[group_col] != reference_group]
        detector = HDDDM(subsets=subsets)
        detector.set_reference(reference_df)
        drift_state = []

        for group_id, subset_data in test_df.groupby(group_col):
            detector.update(subset_data[cols])
            drift_state.append(detector.drift_state)

        return detector, drift_state

    def test_kdq_tree_batch_detector(self, cols, group_col=None, reference_group=None):
        if not group_col:
            group_col = self.categorical_cols[
                random.randint(0, len(self.categorical_cols) - 1)
            ]

            while group_col in cols:
                group_col = self.categorical_cols[
                    random.randint(0, len(self.categorical_cols) - 1)
                ]

        if not reference_group:
            reference_group = self.df[group_col].min()

        reference_df = self.df[self.df[group_col] == reference_group][cols]
        test_df = self.df[self.df[group_col] != reference_group]
        detector = KdqTreeBatch()
        detector.set_reference(reference_df)
        drift_state = []

        for group_id, subset_data in test_df.groupby(group_col):
            detector.update(subset_data[cols])
            drift_state.append(detector.drift_state)

        return detector, drift_state

    def test_kdq_tree_streaming_detector(
        self, cols, window_size=500, alpha=0.05, bootstrap_samples=500, count_ubound=50
    ):
        detector = KdqTreeStreaming(window_size, alpha, bootstrap_samples, count_ubound)
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(row[cols])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def test_lfr_detector(
        self,
        model=None,
        x_cols=None,
        y_col=None,
        time_decay_factor=0.6,
        warning_level=0.01,
        detect_level=0.001,
        num_mc=5000,
        burn_in=10,
        subsample=10,
    ):
        if not model:
            model, x_cols, y_col = self.train_classifier_model(
                model_type="svc", x_cols=x_cols, y_col=y_col, limit_classes=2
            )

        self.df["y_pred"] = model.predict(self.df[x_cols])
        detector = LinearFourRates(
            time_decay_factor=time_decay_factor,
            warning_level=warning_level,
            detect_level=detect_level,
            num_mc=num_mc,
            burn_in=burn_in,
            subsample=subsample,
        )
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(X=row[x_cols], y_true=row[y_col], y_pred=row["y_pred"])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def test_md3_detector(
        self,
        model=None,
        x_cols=None,
        y_col=None,
        start=0,
        end=0.75,
        sensitivity=1.5,
        oracle_labels=1000,
    ):
        if not model:
            model, x_cols, y_col = self.train_classifier_model(
                model_type="svc", x_cols=x_cols, y_col=y_col, start=start, end=end
            )
            retrain_model, _, _ = self.train_classifier_model(
                model_type="svc", x_cols=x_cols, y_col=y_col, start=start, end=end
            )

        end_train = int(end * len(self.df))
        cols = x_cols.copy()
        cols.append(y_col)
        self.df["y_pred"] = model.predict(self.df[x_cols])
        self.df["y_pred_retrain"] = retrain_model.predict(self.df[x_cols])
        detector = MD3(
            clf=model,
            sensitivity=sensitivity,
            oracle_data_length_required=oracle_labels,
        )
        detector.set_reference(X=self.df[cols], target_name=y_col)
        drift_state = []

        for i, row in self.df.iloc[
            end_train : len(self.df),
        ].iterrows():
            if detector.waiting_for_oracle:
                oracle_label = pd.DataFrame([row[cols]])
                detector.give_oracle_label(oracle_label)

                if not detector.waiting_for_oracle:
                    retrain_model.fit(
                        detector.reference_batch_features,
                        detector.reference_batch_target.values.ravel(),
                    )
                    self.df["y_pred_retrain"] = retrain_model.predict(self.df[x_cols])

                drift_state.append(detector.drift_state)
            else:
                detector.update(
                    X=pd.DataFrame([row[x_cols]]),
                    y_true=row[y_col],
                    y_pred=row["y_pred_retrain"],
                )
                drift_state.append(detector.drift_state)

        return detector, drift_state

    def test_nndvi_detector(
        self, cols=None, group_col=None, reference_group=None, k_nn=2, sampling_times=50
    ):
        if not group_col:
            group_col = self.categorical_cols[
                random.randint(0, len(self.categorical_cols) - 1)
            ]

            if cols:
                while group_col in cols:
                    group_col = self.categorical_cols[
                        random.randint(0, len(self.categorical_cols) - 1)
                    ]

        if not reference_group:
            reference_group = self.df[group_col].min()

        filtered_df = self.df.copy()
        for filter_col in filtered_df.columns:
            if filter_col != group_col and not pd.api.types.is_numeric_dtype(
                filtered_df[filter_col]
            ):
                filtered_df.drop(columns=filter_col, inplace=True)

        grouped_df = filtered_df.groupby(group_col)
        status = pd.DataFrame(columns=[group_col, "drift"])
        batches = {
            group_id: group.sample(frac=0.1).drop(columns=group_col).values
            for group_id, group in grouped_df
        }

        detector = NNDVI(k_nn=k_nn, sampling_times=sampling_times)
        detector.set_reference(batches.pop(reference_group))

        for group_id, batch in batches.items():
            detector.update(pd.DataFrame(batch))
            status = pd.concat(
                [
                    status,
                    pd.DataFrame(
                        {group_col: [group_id], "drift": [detector.drift_state]}
                    ),
                ],
                ignore_index=True,
            )

        return detector, status

    def test_pcacd_detector(
        self, cols=None, window_size=50, divergence_metric="intersection"
    ):
        if not cols:
            cols = self.numeric_cols.copy()

        detector = PCACD(window_size=window_size, divergence_metric=divergence_metric)
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(row[cols])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def test_stepd_detector(self, model=None, x_cols=None, y_col=None, window_size=250):
        if not model:
            model, x_cols, y_col = self.train_classifier_model(
                model_type="svc", x_cols=x_cols, y_col=y_col
            )

        self.df["y_pred"] = model.predict(self.df[x_cols])
        detector = STEPD(window_size=window_size)
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(y_true=row[y_col], y_pred=row["y_pred"])
            drift_state.append(detector.drift_state)

        self.df["drift_state"] = drift_state
        return detector

    def plot_drift_scatter(self, cols, output_file="plots/drift_scatter_test.png"):
        plt.figure(figsize=(20, 6))
        y_min = None
        y_max = None

        for col in cols:
            plt.scatter(self.df.index, self.df[col], label=col)
            local_min = self.df[col].min()
            local_max = self.df[col].max()

            if y_min is None or y_min > local_min:
                y_min = local_min
            if y_max is None or y_max < local_max:
                y_max = local_max

        plt.grid(False, axis="x")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("Scatter Results", fontsize=22)
        plt.xlabel("Index", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.ylim((y_min, y_max))
        plt.vlines(
            x=self.df[self.df["drift_state"] == "drift"].index,
            ymin=y_min,
            ymax=y_max,
            label="Drift Detected",
            color="red",
        )
        plt.legend()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
