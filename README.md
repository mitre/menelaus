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

Menelaus is named for the Odyssean hero that defeated the shapeshifting Proteus.

# Detector List

Menelaus implements the following drift detectors.

| Type             | Detector                                                      | Abbreviation | Streaming | Batch |
|------------------|---------------------------------------------------------------|--------------|-----------|-------|
| Change detection | Cumulative Sum Test                                           | CUSUM        | x         |       |
| Change detection | Page-Hinkley                                                  | PH           | x         |       |
| Change detection    | ADaptive WINdowing                                            | ADWIN        | x         |       |
| Concept drift    | Drift Detection Method                                        | DDM          | x         |       |
| Concept drift    | Early Drift Detection Method                                  | EDDM         | x         |       |
| Concept drift    | Linear Four Rates                                             | LFR          | x         |       |
| Concept drift    | Statistical Test of Equal Proportions to Detect concept drift | STEPD        | x         |       |
| Concept drift    | Margin Density Drift Detection Method                         | MD3          | x         |       |
| Data drift       | Confidence Distribution Batch Detection                       | CDBD         |           | x     |
| Data drift       | Hellinger Distance Drift Detection Method                     | HDDDM        |           | x     |
| Data drift       | kdq-Tree Detection Method                                     | kdq-Tree     | x         | x     |
| Data drift       | PCA-Based Change Detection                                    | PCA-CD       | x         |       |
| Ensemble         | Streaming Ensemble      | - | x |
| Ensemble         | Batch Ensemble          | - |   | x |


The three main types of detector are described below. More details, including
references to the original papers, can be found in the respective module
documentation on [ReadTheDocs](https://menelaus.readthedocs.io/en/latest/).

-   Change detectors monitor single variables in the streaming context,
    and alarm when that variable starts taking on values outside of a
    pre-defined range.
-   Concept drift detectors monitor the performance characteristics of a
    given model, trying to identify shifts in the joint distribution of
    the data\'s feature values and their labels. Note that change detectors 
    can also be applied in this context.
-   Data drift detectors monitor the distribution of the features; in
    that sense, they are model-agnostic. Such changes in distribution
    might be to single variables or to the joint distribution of all the
    features.
-   Ensembles are groups of detectors, where each watches the same data, and 
    drift is determined by combining their output. Menelaus implements a 
    framework for wrapping detectors this way.

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
cd ./menelaus_clone_folder/
pip install -e .[dev] 
```

Menelaus should work with Python 3.8 or higher. 

# Getting Started

Each detector implements the API defined by `menelaus.detector`:
notably, they have an `update` method which allows new data to be passed, and a `drift_state` attribute which tells the user whether drift has been
detected, along with (usually) other attributes specific to the detector class.

Generally, the workflow for using a detector, given some data, is as
follows:

```python
from menelaus.concept_drift import ADWINAccuracy
from menelaus.data_drift import KdqTreeStreaming
from menelaus.datasets import fetch_rainfall_data
from menelaus.ensemble import StreamingEnsemble, SimpleMajorityElection


# has feature columns, and a binary response 'rain'
df = fetch_rainfall_data()


# use a concept drift detector (response-only)
detector = ADWINAccuracy()
for i, row in df.iterrows():
    detector.update(X=None, y_true=row['rain'], y_pred=0)
    assert detector.drift_state != "drift", f"Drift detected in row {i}"


# use data drift detector (features-only)
detector = KdqTreeStreaming(window_size=5)
for i, row in df.iterrows():
    detector.update(X=df.loc[[i], df.columns != 'rain'], y_true=None, y_pred=None)
    assert detector.drift_state != "drift", f"Drift detected in row {i}"


# use ensemble detector (detectors + voting function)
ensemble = StreamingEnsemble(
  {
    'a': ADWINAccuracy(),
    'k': KdqTreeStreaming(window_size=5)
  },
  SimpleMajorityElection()
)

for i, row in df.iterrows():
    ensemble.update(X=df.loc[[i], df.columns != 'rain'], y_true=row['rain'], y_pred=0)
    assert ensemble.drift_state != "drift", f"Drift detected in row {i}"
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
  bandit -r ./menelaus        # security checks
  black ./menelaus   # formatting
  ```  

# Copyright

Authors: Leigh Nicholl, Thomas Schill, India Lindsay, Anmol Srivastava, Kodie P McNamara, Shashank Jarmale.\
©2022 The MITRE Corporation. ALL RIGHTS RESERVED\
Approved for Public Release; Distribution Unlimited. Public Release\
Case Number 22-0244.
