[![tests](https://github.com/mitre/menelaus/actions/workflows/tests.yml/badge.svg)](https://github.com/mitre/menelaus/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/menelaus/badge/?version=latest)](https://menelaus.readthedocs.io/en/latest/?badge=latest)
[![examples](https://github.com/mitre/menelaus/actions/workflows/examples.yml/badge.svg?branch=main)](https://github.com/mitre/menelaus/actions/workflows/examples.yml)
[![lint](https://github.com/mitre/menelaus/actions/workflows/format.yml/badge.svg)](https://github.com/mitre/menelaus/actions/workflows/format.yml)

# Background

Menelaus implements algorithms for the purposes of drift detection. Drift
detection is a branch of machine learning focused on the detection of unforeseen
shifts in data. The relationships between variables in a dataset are rarely
static and can be affected by changes in both internal and external factors,
e.g. changes in data collection techniques, external protocols, and/or
population demographics. Both undetected changes in data and undetected model
underperformance pose risks to the users thereof. The aim of this package is to
enable monitoring of data and of model performance.

The algorithms contained within this package were identified through a
comprehensive literature survey. Menelaus\' aim was to implement drift detection
algorithms that cover a range of statistical methodology. Of the algorithms
identified, all are able to identify when drift is occurring; some can highlight
suspicious regions of the data in which drift is more significant; and others
can also provide model retraining recommendations.

Menelaus implements drift detectors for both streaming and batch data. In a
streaming setting, data is arriving continuously and is processed one
observation at a time. Streaming detectors process the data with each new
observation that arrives and are intended for use cases in which instant
analytical results are desired. In a batch setting, information is collected
over a period of time. Once the predetermined set is \"filled\", data is fed
into and processed by the drift detection algorithm as a single batch. Within a
batch, there is no meaningful ordering of the data with respect to time. Batch
algorithms are typically used when it is more important to process large volumes
of information simultaneously, where the speed of results after receiving data
is of less concern.

In The Odyssey, Menelaus seeks a prophecy known by the shapeshifter
Proteus. Menelaus holds Proteus down as he takes the form of a lion, a
serpent, water, and so on. Eventually, Proteus relents, and Menelaus
gains the answers he sought. Accordingly, this library provides tools
for \"holding\" data as it shifts.

# Detector List

Menelaus implements the following drift detectors.

| Type             | Detector                                                      | Abbreviation | Streaming | Batch |
|------------------|---------------------------------------------------------------|--------------|-----------|-------|
| Change detection | Cumulative Sum Test                                           | CUSUM        | x         |       |
| Change detection | Page-Hinkley                                                  | PH           | x         |       |
| Concept drift    | ADaptive WINdowing                                            | ADWIN        | x         |       |
| Concept drift    | Drift Detection Method                                        | DDM          | x         |       |
| Concept drift    | Early Drift Detection Method                                  | EDDM         | x         |       |
| Concept drift    | Linear Four Rates                                             | LFR          | x         |       |
| Concept drift    | Statistical Test of Equal Proportions to Detect concept drift | STEPD        | x         |       |
| Concept drift    | Margin Density Drift Detection Method                         | MD3          | x         |       |
| Data drift       | Confidence Distribution Batch Detection                       | CDBD         |           | x     |
| Data drift       | Hellinger Distance Drift Detection Method                     | HDDDM        |           | x     |
| Data drift       | kdq-Tree Detection Method                                     | kdq-Tree     | x         | x     |
| Data drift       | PCA-Based Change Detection                                    | PCA-CD       | x         |       |


The three main types of detector are described below. More details, including
references to the original papers, can be found in the respective module
documentation on [ReadTheDocs](https://menelaus.readthedocs.io/en/latest/).

-   Change detectors monitor single variables in the streaming context,
    and alarm when that variable starts taking on values outside of a
    pre-defined range.
-   Concept drift detectors monitor the performance characteristics of a
    given model, trying to identify shifts in the joint distribution of
    the data\'s feature values and their labels.
-   Data drift detectors monitor the distribution of the features; in
    that sense, they are model-agnostic. Such changes in distribution
    might be to single variables or to the joint distribution of all the
    features.

The detectors may be applied in two settings, as described in the Background
section:

-   Streaming, in which each new observation that arrives is processed
    separately, as it arrives.
-   Batch, in which the data has no meaningful ordering with respect to time,
    and the goal is comparing two datasets as a whole.

Additionally, the library implements a kdq-Tree partitioner, for support of the
kdq-Tree Detection Method. This data structure partitions a given feature space,
then maintains a count of the number of samples from the given dataset that fall
into each section of that partition. More details are given in the respective
module.

# Installation

Create a virtual environment as desired, then:

```python
# for read-only, install from pypi:
pip install menelaus

# to allow editing, running tests, generating docs, etc.
# First, clone the git repo, then:
cd ./menelaus/
pip install -e .[dev] 
```

Menelaus should work with Python 3.8 or higher. 

# Getting Started

Each detector implements the API defined by `menelaus.drift_detector`:
they have an `update` method which allows new data to be passed, a
`drift_state` attribute which tells the user whether drift has been
detected, and a `reset` method (generally called automatically by
`update`) which clears the `drift_state` along with (usually) some other
attributes specific to the detector class.

Generally, the workflow for using a detector, given some data, is as
follows:

```python
import pandas as pd
from menelaus.concept_drift import ADWINOutcome
from menelaus.data_drift import KdqTreeStreaming

df = pd.read_csv('example.csv')

# use a detector that searches for concept drift
detector = ADWINOutcome()
for i, row in df.iterrows():
   detector.update(row['y_true'], row['y_predicted'], X=None)
   if detector.drift_state is not None:
      print("Drift has occurred!")

# use a detector that searches for data drift
detector = KdqTreeStreaming()
feature_cols = ['a', 'b', 'c']
for i in range(len(df)):
   detector.update(X=df.loc[[i], feature_cols], y_true=None, y_pred=None)
   if detector.drift_state is not None:
      print("Drift has occurred!")
```

As a concept drift detector, ADWIN requires both a true value (`y_true`) and a
predicted value (`y_predicted`) at each update step. The data drift detector
KdqTreeStreaming only requires the feature values at each step (`X`). More
detailed examples, including code for visualizating drift locations, may be
found in the ``examples`` directory, as stand-alone python scripts. The examples
along with output can also be viewed on the RTD website.

# Contributing
Install the library using the `[dev]` option, as above.

- **Testing**

  Unit tests can be run with the command `pytest`. By default, a
  coverage report with highlighting will be generated in `htmlcov/index.html`.
  These default settings are specified in `setup.cfg` under `[tool:pytest]`.

- **Documentation**

  HTML documentation can be generated at
  `menelaus/docs/build/html/index.html` with:
  ```python
  cd docs/source
  sphinx-build . ../build
  ```

  If the example notebooks for the docs need to be updated, the corresponding 
  python scripts in the `examples` directory should also be regenerated via:
  ```python
  cd docs/source/examples
  python convert_notebooks.py
  ```
  Note that this will require the installation of `jupyter` and `nbconvert`,
  which can be added to installation via `pip install -e ".[dev, test]"`.

- **Formatting**:

  This project uses `black`, `bandit`, and `flake8` for code formatting and
  linting, respectively. To satisfy these requirements when contributing, you
  may use them as the linter/formatter in your IDE, or manually run the
  following from the root directory:
  ```python
  flake8                 # linting
  bandit -r ./src        # security checks
  black ./src/menelaus   # formatting
  ```  

# Copyright

Authors: Leigh Nicholl, Thomas Schill, India Lindsay, Anmol Srivastava, Kodie P McNamara, Shashank Jarmale.\
??2022 The MITRE Corporation. ALL RIGHTS RESERVED\
Approved for Public Release; Distribution Unlimited. Public Release\
Case Number 22-0244.
