''' This is an example script to generate a mock plot for the benchmarks/
    README. It uses detectors from the package to compare algorithm
    efficacy on example data.
'''

import matplotlib.pyplot as plt

from menelaus.data_drift import KdqTreeBatch, KdqTreeStreaming
from menelaus.datasets import make_example_batch_data


# setup data
data = make_example_batch_data()
data_batches = [x for _,x in data.groupby('year')]
data_batches = [x[[c for c in 'abcdefghij']] for x in data_batches]
data = data[[c for c in 'abcdefghij']]
batch_size = len(data) // len(data_batches)

# set up detectors 
kdq_streaming = KdqTreeStreaming(window_size=100)
kdq_batch = KdqTreeBatch()

# run
batch_det_locs = []
for i in range(len(data_batches)):
    kdq_batch.update(data_batches[i])
    if kdq_batch.drift_state:
        print('Batch detector drift state', kdq_batch.drift_state)
        batch_det_locs.append((i+1)*batch_size)

stream_det_locs = []
for i in range(len(data)):
    kdq_streaming.update(data.iloc[[i]])
    if kdq_streaming.drift_state:
        print('Stream detector drift state', kdq_streaming.drift_state)
        stream_det_locs.append(i)

# plot
f = plt.figure(figsize=(20,6))
plt.grid(False, axis='x')
plt.xticks(fontsize=16)
plt.title('KdqTreeBatch vs. KdqTreeStreaming on Example Data', fontsize=22)
plt.ylabel('', fontsize=18)
plt.xlabel('index', fontsize=18)
plt.vlines(batch_det_locs, ymin=0, ymax=10, label='KdqTreeBatch alarm', color='red')
plt.vlines(stream_det_locs, ymin=0, ymax=10, label='KdqTreeStreaming alarm', color='purple')
plt.legend()
plt.savefig('figures/bm_example_plot_1.png')
