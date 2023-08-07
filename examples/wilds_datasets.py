#!/usr/bin/env python
# coding: utf-8

# #### Overview 
# 
# This notebook is a work in progress. Eventually, the contents will demonstrate an NLP-based drift detection algorithm in action, but until the feature is developed, it shows the loading and use of two datasets to be used in the examples:
# 
# - Civil Comments dataset: online comments to be used in toxicity classification problems 
# - Amazon Reviews dataset: amazon reviews to be used in a variety of NLP problems
# 
# The data is accessed by using the `wilds` library, which contains several such datasets and wraps them in an API as shown below. 
# 
# #### Imports

# In[4]:


from wilds import get_dataset


# #### Load Data
# 
# Note that initially, the large data files need to be downloaded first. Later examples may assume the data is already stored to disk.

# In[7]:


# amazon reviews
dataset_amazon = get_dataset(dataset="amazon", download=True)
dataset_amazon


# In[8]:


# civil comments
dataset_civil = get_dataset(dataset="civilcomments", download=True)
dataset_civil


# In[ ]:




