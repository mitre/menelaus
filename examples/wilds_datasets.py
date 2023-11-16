#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# This example demonstrates an experimental NLP-based drift detection algorithm. It uses the "Civil Comments" dataset ([link](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/civilcomments_dataset.py) to a Python loading script with additional details/links) from the `wilds` library, which contains online comments meant to be used in toxicity classification problems.
# 
# This example and the experimental modules often pull directly and indirectly from [`alibi-detect`](https://github.com/SeldonIO/alibi-detect/tree/master) and its own [example(s)](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_text_imdb.html).
# 
# ## Notes
# 
# This code is experimental, and has notable issues:
# - transform functions are very slow, on even moderate batch sizes
# - detector design is not generalized, and may not work on streaming problems, or with data representations of different types/shapes
# - some warnings below are not addressed
# - if not present, `toolz`, `tensorflow`, and `transformers` must be added via the `experimental` install option, and are not included by default
# 
# ## Imports
# 
# Code (transforms, alarm, detector) is pulled from the experimental module in `menelaus`, which is live but not fully tested. Note that commented code shows `wilds` modules being used to access and save the dataset to disk, but are excluded to save time. The example hence assumes the dataset is locally available.

# In[3]:


import pickle
# import pandas as pd
# from wilds import get_dataset

from menelaus.experimental.transform import auto_tokenize, extract_embedding, uae_reduce_dimension
from menelaus.experimental.detector import Detector
from menelaus.experimental.alarm import KolmogorovSmirnovAlarm


# ## Load Data
# 
# Since some of the experimental modules are not very performant, the dataset is loaded and then limited to the first 300 data points (comments), which are split into three sequential batches of 100.
# 
# __Note__: for convenience in generating documentation, the sample is itself saved locally and read from disk in the below examples, but the commented code describes the steps. 

# In[4]:


# civil comments
# dataset_civil = get_dataset(dataset="civilcomments", download=True, root_dir="./wilds_datasets")
# dataset_civil = pd.read_csv('wilds_datasets/civilcomments_v1.0/all_data_with_identities.csv')
# dataset_civil = dataset_civil['comment_text'][:300].tolist()

# with open('civil_comments_sample.pkl', 'wb') as f:
#     pickle.dump(dataset_civil, f)

dataset_civil = None
with open('civil_comments_sample.pkl', 'rb') as f:
    dataset_civil = pickle.load(f)

batch1 = dataset_civil[:100]
batch2 = dataset_civil[100:200]
batch3 = dataset_civil[200:300]


# ## Transforms Pipeline
# 
# The major step is to initialize the transform functions that will be applied to the comments, to turn them into detector-compatible representations. 
# 
# First, the comments must be tokenized:
# - set up an `AutoTokenizer` model from the `transformers` library with a convenience function, by specifying the desired model name and other arguments
# - the convenience function lets the configured tokenizer be called repeatedly, using batch 1 as the training data
# 
# Then, the tokens must be made into embeddings:
# - an initial transform function uses a `transformers` model to extract embeddings from given tokens
# - the subsequent transform function reduces the dimension via an `UntrainedAutoEncoder` to a manageable size

# In[5]:


# tokens 
tokenizer = auto_tokenize(model_name='bert-base-cased', padding='longest', return_tensors='tf')
tokens = tokenizer(data=batch1)

# embedding (TODO abstract this layers line)
layers = [-_ for _ in range(1, 8 + 1)]
embedder = extract_embedding(model_name='bert-base-cased', embedding_type='hidden_state', layers=layers)

# dimension reduction via Untrained AutoEncoder
uae_reduce = uae_reduce_dimension(enc_dim=32)


# ## Detector Setup
# 
# Next a detector is setup. First, a `KolmogorovSmirnovAlarm` is initialized with default settings. When the amount of columns (which reject the null KS test hypothesis) exceeds the default ratio (0.25), this alarm will indicate drift has occurred. 
# 
# Then the detector is constructed. It is given the initialized alarm, and the ordered list of transforms configured above. The detector is then made to step through each available batch, and its state is printed as output. Note that the first batch establishes the reference data, the second establishes the test data, and the third will require recalibration (test is combined into reference) if drift is detected.

# In[6]:


# detector + set reference
ks_alarm = KolmogorovSmirnovAlarm()
detector = Detector(alarm=ks_alarm, transforms=[tokenizer, embedder, uae_reduce])
detector.step(batch1)
print(f"\nState after initial batch: {detector.state}\n")

# detector + add test   
detector.step(batch2)
print(f"\nState after test batch: {detector.state}\n")

# recalibrate and re-evaluate (XXX - all batches must be same length)
detector.step(batch3)
print(f"\nState after new batch, recalibration: {detector.state}\n")


# ## Final Notes
# 
# We can see the baseline state after processing the initial batch, an alarm raised after observing test data, and then another alarm signal after a new test batch is observed and the reference is internally recalibrated.
