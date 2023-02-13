import matplotlib.pyplot as plt
import os
import pandas as pd
import random
from scipy.io.arff import loadarff

from menelaus.concept_drift import LinearFourRates, ADWINAccuracy, DDM, EDDM, STEPD, MD3
from menelaus.data_drift import PCACD, KdqTreeStreaming, KdqTreeBatch, NNDVI
import class_manipulation
import feature_manipulation
import noise


def select_random_classes(series):
    classes = series.unique()

    if len(classes) < 2:
        raise ValueError(f'Insufficient classes in series: {len(classes)}')
    else:
        class_a = classes[random.randint(0, len(classes) - 1)]
        class_b = classes[random.randint(0, len(classes) - 1)]

        while class_a == class_b:
            class_b = classes[random.randint(0, len(classes) - 1)]

        return [class_a, class_b]


class InjectionTesting:
    def __init__(self, data_path, seed=None, numeric_cols=None, categorical_cols=None):
        file_type = data_path.split('.')[-1]
        self.numeric_cols = []
        self.categorical_cols = []

        if file_type == 'csv':
            self.df = pd.read_csv(data_path)
        elif file_type == 'arff':
            raw_data = loadarff(data_path)
            self.df = pd.DataFrame(raw_data[0])
        else:
            raise ValueError(f'Invalid file type: {file_type}')

        if numeric_cols is None or categorical_cols is None:
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]) and numeric_cols is None:
                    self.numeric_cols.append(col)
                elif self.df[col].nunique() < len(self.df) and categorical_cols is None:
                    self.categorical_cols.append(col)

        if seed:
            random.seed(seed)


    def select_rows(self, start, end):
        start_drift = int(start * len(self.df))
        end_drift = int(end * len(self.df))

        return [start_drift, end_drift]


    def inject_random_brownian_noise(self, x, start=.75, end=1, num_drift_cols=1):
        rand_cols = []
        start_drift, end_drift = self.select_rows(start, end)

        for i in range(num_drift_cols):
            rand_col = self.numeric_cols[random.randint(0, len(self.numeric_cols) - 1)]
            rand_cols.append(rand_col)

            self.df = noise.brownian_noise(self.df, rand_col, x, start_drift, end_drift)

        return rand_cols


    def inject_random_class_manipulation(self, manipulation_type, start=.75, end=1, num_drift_cols=1):
        rand_cols = []
        all_rand_classes = []
        start_drift, end_drift = self.select_rows(start, end)

        for i in range(num_drift_cols):
            rand_col = self.categorical_cols[random.randint(0, len(self.categorical_cols) - 1)]
            rand_cols.append(rand_col)
            rand_classes = select_random_classes(self.df[rand_col])
            all_rand_classes.append(rand_classes)

            if manipulation_type == 'class_swap':
                self.df = class_manipulation.class_swap(self.df, rand_col, rand_classes[0], rand_classes[1], start_drift, end_drift)
            elif manipulation_type == 'class_join':
                new_label = f'{rand_classes[0]}_{rand_classes[1]}'
                self.df = class_manipulation.class_join(self.df, rand_col, rand_classes[0], rand_classes[1], new_label, start_drift, end_drift)
            else:
                raise ValueError(f'Invalid class manipulation type: {manipulation_type}')

        return rand_cols, all_rand_classes


    def inject_random_feature_swap(self, start=.75, end=1, num_swaps=1):
        all_swap_cols = []
        start_drift, end_drift = self.select_rows(start, end)

        for i in range(num_swaps):
            col_type = self.numeric_cols if random.randint(0, 1) == 0 else self.categorical_cols

            if len(col_type) < 2:
                col_type = self.numeric_cols if col_type == self.categorical_cols else self.categorical_cols
            if len(col_type) < 2:
                raise ValueError('Insufficient numeric and categorical columns for swaps')

            col_a = col_type[random.randint(0, len(col_type) - 1)]
            col_b = col_type[random.randint(0, len(col_type) - 1)]

            while col_a == col_b:
                col_b = col_type[random.randint(0, len(col_type) - 1)]

            swap_cols = [col_a, col_b]
            all_swap_cols.append(swap_cols)
            self.df = feature_manipulation.feature_swap(self.df, col_a, col_b, start_drift, end_drift)

        return all_swap_cols


    def inject_random_feature_hide_and_sample(self):
        rand_col = self.df.columns[random.randint(0, len(self.df.columns) - 1)]
        sample_size = min(self.df[rand_col].value_counts())
        self.df = feature_manipulation.feature_hide_and_sample(self.df, rand_col, sample_size)

        return rand_col


    def test_adwin_detector(self, cols):
        detector = ADWINAccuracy()
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(X=None, y_true=row[cols], y_pred=0)
            drift_state.append(detector.drift_state)

        self.df['drift_state'] = drift_state


    def test_kdq_tree_streaming_detector(self, cols, window_size=500, alpha=0.05, bootstrap_samples=500, count_ubound=50):
        detector = KdqTreeStreaming(window_size, alpha, bootstrap_samples, count_ubound)
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(row[cols])
            drift_state.append(detector.drift_state)

        self.df['drift_state'] = drift_state


    def test_nndvi_detector(self, cols, group_name=None, k_nn=2, sampling_times=50):
        if not group_name:
            group_name = self.categorical_cols[random.randint(0, len(self.categorical_cols) - 1)]

            while group_name in cols:
                group_name = self.categorical_cols[random.randint(0, len(self.categorical_cols) - 1)]

        filtered_df = self.df.copy()
        for filter_col in filtered_df.columns:
            if filter_col != group_name and not pd.api.types.is_numeric_dtype(filtered_df[filter_col]):
                filtered_df.drop(columns=filter_col, inplace=True)

        grouped_df = filtered_df.groupby(group_name)
        status = pd.DataFrame(columns=[group_name, 'drift'])
        batches = {group_id: group.sample(frac=0.1).drop(columns=group_name).values for group_id, group in grouped_df}

        detector = NNDVI(k_nn=k_nn, sampling_times=sampling_times)
        detector.set_reference(batches.pop(min(self.df[group_name])))

        for group_id, batch in batches.items():
            detector.update(pd.DataFrame(batch))
            status = pd.concat([status, pd.DataFrame({group_name: [group_id], 'drift': [detector.drift_state]})], ignore_index=True)

        return status


    def test_pcacd_detector(self, window_size=50, divergence_metric='intersection'):
        detector = PCACD(window_size=window_size, divergence_metric=divergence_metric)
        drift_state = []

        for i, row in self.df.iterrows():
            detector.update(row)
            drift_state.append(detector.drift_state)

        self.df['drift_state'] = drift_state


    def plot_drift_scatter(self, cols, output_file='plots/drift_scatter_test.png'):
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

        plt.grid(False, axis='x')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Scatter Results', fontsize=22)
        plt.xlabel('Index', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.ylim((y_min, y_max))
        plt.vlines(x=self.df[self.df['drift_state'] == 'drift'].index, ymin=y_min, ymax=y_max, label='Drift Detected', color='red')
        plt.legend()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)


if __name__ == '__main__':
    file = 'souza_data/INSECTS-abrupt_balanced_norm.arff'
    tester = InjectionTesting(file)
    drift_cols = tester.inject_random_brownian_noise(10)
    tester.test_nndvi_detector(drift_cols)
