"""
NNDVI Example -- currently used to debug false positives issue #36.
"""

import os
import numpy as np
import pandas as pd 

from menelaus.data_drift import NNDVI

data = pd.read_csv(
    os.path.join("src", "menelaus", "tools", "artifacts", "example_data.csv"),
    index_col="id"
)
data_grouped = data.groupby('year')
batches = [	group 
			for _, group in data_grouped ]
batches = [	batch.drop(['year', 'cat', 'confidence', 'drift'], axis=1)
			for batch in batches ]
# later, also group/store the response values
batches = [ np.array(batch.values)
			for batch in batches ]


nndvi = NNDVI(k_nn=2, sampling_times=50)
nndvi.set_reference(batches[0])

for batch in batches[1:]:
	nndvi.update(batch)
	print(f"drift state = {nndvi.drift_state}")