#!/usr/bin/env python
# coding: utf-8

# # Ensemble Drift Detector Examples
# 
# This notebook contains examples on how to build and use ensemble detectors using the individual algorithms in the `menelaus` suite. These examples also include instructions on specifying evaluation schemes and setting custom subsets of data per constituent detector.  
# 
# Most parameterizations and initalizations therein may not result in optimal performance or detection, and are provided just for demonstration.
# 
# ## Imports

# In[1]:


import numpy as np

from menelaus.concept_drift import STEPD
from menelaus.datasets import make_example_batch_data, fetch_rainfall_data
from menelaus.data_drift import HDDDM, KdqTreeBatch, KdqTreeStreaming
from menelaus.ensemble import BatchEnsemble, StreamingEnsemble
from menelaus.ensemble import SimpleMajorityElection, MinimumApprovalElection


# ## Import Data

# In[2]:


example_data = make_example_batch_data()
rainfall_data = fetch_rainfall_data()


# ## Batch Ensemble
# 
# The simplest use of an ensemble is to combine three data-drift-only detectors with few additional settings. In this case we can combine three instances of batch detectors (`KdqTreeBatch`, `HDDDM`), all operating on the same data columns, with a very basic evaluation scheme (*i.e.* a simple majority of detectors alarming, causes the ensemble to alarm).

# In[3]:


# initialize set of detectors with desired parameterizations
detectors = {
    'k1': KdqTreeBatch(bootstrap_samples=500),
    'k2': KdqTreeBatch(bootstrap_samples=475),
    'h1': HDDDM()
}

# choose an election scheme
election = SimpleMajorityElection()

# initialize an ensemble object
ensemble = BatchEnsemble(detectors, election)


# Note that `BatchEnsemble` and `StreamingEnsemble` are instances of `BatchDetector` and `StreamingDetector` themselves (respectively). As such, they are used in the same syntactic way and possess similar properties.

# In[4]:


# make dataset smaller
df_example_data = example_data[example_data.year < 2010]

# split dataset into 1 dataset for each 'batch' (year)
df_into_batches = [x for _,x in df_example_data.groupby('year')]
df_into_batches = [x[['a', 'b', 'c']] for x in df_into_batches]

# batch detectors -- and ensembles -- need an initial reference batch
ensemble.set_reference(df_into_batches[0])
print(f"Batch #{0} | Ensemble reference set")

for i, batch in enumerate(df_into_batches[1:]):
    ensemble.update(batch)
    print(f"Batch #{i+1} | Ensemble overall drift state: {ensemble.drift_state}")


# ## Streaming Ensemble
# 
# Using an ensemble of streaming detectors can involve additional features. This example uses both data and concept drift detectors (`KdqTreeStreaming`, `STEPD`), custom subsets of data for different detectors, as well as a different election scheme that will alarm if a custom, minimum number of detectors "approve" or alarm for drift.

# In[5]:


# initialize set of detectors with desired parameterizations
detectors = {
    'k1': KdqTreeStreaming(window_size=200, bootstrap_samples=250),
    'k2': KdqTreeStreaming(window_size=225, bootstrap_samples=200),
    's1': STEPD(window_size=50)
}

# functions that select the part of 'X' each detector needs - keys must match!
column_selectors = {
    'k1': lambda x: x[['temperature', 'visibility', 'dew_point']],
    'k2': lambda x: x[['temperature', 'visibility', 'average_wind_speed']]
}

# choose an election scheme
election = MinimumApprovalElection(approvals_needed=1)

# initialize an ensemble object
stream_ensemble = StreamingEnsemble(detectors, election, column_selectors)


# When mixing concept and data drift detectors, it's especially important to pass data explicitly.

# In[6]:


# make data smaller
df_stream = rainfall_data[0:1000]

# random "predicted" outcomes -- in case a concept drift detector needs them
y_preds = np.random.randint(low=0, high=2, size=1000)

# use ensemble
for i, row in df_stream.iterrows():
    stream_ensemble.update(
        X=df_stream.loc[[i]],
        y_true=row['rain'],
        y_pred=y_preds[i]
    )
    if stream_ensemble.drift_state is not None:
        print(f"Example #{i} | Ensemble drift state: {stream_ensemble.drift_state}")

