|pipeline| |coverage|

.. |pipeline| image:: https://gitlab.mitre.org/lnicholl/molten/badges/dev/pipeline.svg
   :target: https://gitlab.mitre.org/lnicholl/molten/-/commits/dev

.. |coverage| image:: https://gitlab.mitre.org/lnicholl/molten/badges/dev/coverage.svg
   :target: https://gitlab.mitre.org/lnicholl/molten/-/commits/dev


Background
==========

Menelaus implements algorithms for the purposes of drift detection. Drift
detection is a branch of machine learning focused on the real-time detection of
unforeseen shifts in data. The relationships between variables in a dataset are
sensitive and rarely static and can be affected by changes in both internal and
external factors. These factors include changes in data collection techniques,
population demographics, and external protocols. 
 
When drift occurs, the data to which a model is being applied can differ
statistically from the data the model was trained on. This may lead to a
decrease in both the discrimination and calibration of a deployed machine
learning model or even of a rule-based system learned on the data. The goal of
drift detection algorithms is to detect a change in either a model's error rate
or in the distribution of features within a dataset. 
 
Both undetected changes in data and undetected model underperformance pose risks
to the users thereof. The aim of this package is to enable monitoring of data
and machine learning models. 
 
The algorithms contained within this package were identified through a
comprehensive literature survey. Menelaus' aim was to implement drift detection
algorithms that cover a diverse range of statistical methodology. Of the
algorithms identified, all are able to identify when drift is occurring; some
can highlight suspicious regions of the dataspace in which drift is more
significant; and others can also provide model retraining recommendations. 
 
Menelaus implements drift detectors for both streaming and batch data. In a
streaming setting, data is arriving continuously and is processed one
observation at a time. Streaming detectors process the data with each new
observation that arrives and are intended for use cases in which instant
analytical results are desired. In a batch setting, information is collected
over a period of time. Once the predetermined set is "filled", data is fed into
and processed by the drift detection algorithm as a single batch. Within a
batch, there is no meaningful ordering of the data with respect to time. Batch
algorithms are typically used when it is more important to process large volumes
of information simultaneously, where the speed of results after receiving data
is of less concern.


Detector List
============================

Menelaus implements the following drift detectors.

+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Type              | Detector                                                       | Abbreviation  | Streaming  | Batch  |
+===================+================================================================+===============+============+========+
| Change detection  | Cumulative Sum Test                                            | CUSUM         | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Change detection  | Page-Hinkley                                                   | PH            | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Concept drift     | ADaptive WINdowing                                             | ADWIN         | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Concept drift     | Drift Detection Method                                         | DDM           | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Concept drift     | Early Drift Detection Method                                   | EDDM          | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Concept drift     | Linear Four Rates                                              | LFR           | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Concept drift     | Statistical Test of Equal Proportions to Detect concept drift  | STEPD         | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Data drift        | Confidence Distribution Batch Detection                        | CDBD          |            | x      |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Data drift        | Hellinger Distance Drift Detection Method                      | HDDDM         |            | x      |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Data drift        | kdq-Tree Detection Method                                      | kdq-Tree      | x          | x      |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+
| Data drift        | PCA-Based Change Detection                                     | PCA-CD        | x          |        |
+-------------------+----------------------------------------------------------------+---------------+------------+--------+

The three main types of detector are described below. More details can be found 
in the respective module documentation:

* Change detectors monitor single variables in the streaming context, and alarm 
  when that variable starts taking on values outside of a pre-defined range.

* Concept drift detectors monitor the performance characteristics of a given
  model, trying to identify shifts in the joint distribution of the data's
  feature values and their labels.

* Data drift detectors monitor the distribution of the features; in that sense,
  they are model-agnostic. Such changes in distribution might be to single
  variables or to the joint distribution of all the features.

The detectors may be applied in two settings, as described previously in the
Background section:

* Streaming, in which each new observation that arrives is processed separately,
  as it arrives.

* Batch, in which the data has no meaningful ordering with respect to time, and
  the goal is comparing two datasets as a whole.

Additionally, the library implements a kdq-Tree partitioner, for support of the
kdq-Tree Detection Method. This data structure partitions a given feature space,
then maintains a count of the number of samples from the given dataset that fall
into each section of that partition. More details are given in the respective
module.



Installation
============================

Create a virtual environment as desired, e.g. ``python -m venv ./venv``, then:

.. code-block:: python

   cd ./menelaus/
   
   #for read-only:
   pip install . 

   #to allow editing, running tests, generating docs, etc.
   pip install -e .[dev] 

Menelaus will generally work with Python 3.7 or higher; more specific version
testing is in the works.

Getting Started
============================
Each detector implements the API defined by ``menelaus.drift_detector``: they
have an ``update`` method which allows new data to be passed, a ``drift_state``
attribute which tells the user whether drift has been detected, and a ``reset``
method (generally called automatically by ``update``) which clears the
``drift_state`` along with (usually) some other attributes specific to the 
detector class.

Generally, the workflow for using a detector, given some data, is as follows:

.. code-block:: python

   import pandas as pd
   from menelaus.concept_drift import ADWIN
   df = pd.read_csv('example.csv')
   detector = ADWIN()
   for i, row in df.iterrows():
      detector.update(row['y_predicted'], row['y_true'])
      if detector.drift_state is not None:
         print("Drift has occurred!")

For this example, because ADWIN is a concept drift detector, it requires both a
predicted value (``y_predicted``) and a true value (``y_true``), at each update
step. Note that this requirement is not true for the detectors in other modules.
More detailed examples, including code for visualizating drift locations, may be
found in the ``examples`` directory, as stand-alone python scripts.


Testing and Documentation
============================

After installation using the ``[dev]`` option above, unit tests can be run and 
and html documentation can be generated.

Unit tests can be run with the command ``pytest``. By default, a coverage 
report with highlighting will be generated in ``htmlcov/index.html``. These
default settings are specified in ``setup.cfg`` under ``[tool:pytest]``.

HTML documentation can be generated at ``menelaus/docs/build/html/index.html`` with:

.. code-block:: python

   cd docs
   sphinx-apidoc -M --templatedir source/templates -f -o source ../src/menelaus && make clean && make html




Copyright
============================
| Authors: Leigh Nicholl, Thomas Schill, India Lindsay, Anmol Srivastava, Kodie P McNamara, Austin Downing.
| ©2022 The MITRE Corporation. ALL RIGHTS RESERVED
| Approved for Public Release; Distribution Unlimited. Public Release Case Number 22-0244.
