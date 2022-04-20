|pipeline| |coverage|

.. |pipeline| image:: https://gitlab.mitre.org/lnicholl/molten/badges/dev/pipeline.svg
   :target: https://gitlab.mitre.org/lnicholl/molten/-/commits/dev

.. |coverage| image:: https://gitlab.mitre.org/lnicholl/molten/badges/dev/coverage.svg
   :target: https://gitlab.mitre.org/lnicholl/molten/-/commits/dev


Background
==========

MOLTEN (MOdel Longevity Test ENgine) implements algorithms for the purposes of drift detection. Drift detection is a branch of machine learning focused on the real-time detection of unforeseen shifts in data. The relationships between variables in a dataset are sensitive and rarely static and can be affected by changes in both internal and external factors. These factors include changes in data collection techniques, population demographics, and external protocols. 
 
When drift occurs, the data to which a model is being applied can differ statistically from the data the model was trained on. This may lead to a decrease in both the discrimination and calibration of a deployed machine learning model, or even a rule-based system learned on the data. The goal of drift detection algorithms is to detect a change in either a model's error rate or in the distribution of feature variables within a dataset. 
 
Both undetected changes in data and undetected model underperformance pose risks to the users thereof. The aim of this package is to enable monitoring of data and machine learning models. 
 
The algorithms contained within this package were identified through a comprehensive literature survey. MOLTEN's aim was to implement drift detection algorithms that cover a diverse range of statistical methodology. Of the algorithms identified, all are able to identify when drift is occurring; some can highlight suspicious regions of the dataspace in which drift is more significant; and others can also provide model retraining recommendations. 
 
MOLTEN implements drift detectors for both streaming and batch data. In a streaming setting, data is arriving continuously and is processed one observation at a time. Streaming detectors process the data with each new observation that arrives and are intended for use cases in which instant analytical results are desired. In a batch setting, information is collected over a period of time. Once the predetermined set is "filled", data is fed into and processed by the drift detection algorithm as a single batch. Within a batch, there is no meaningful ordering of the data with respect to time. Batch algorithms are typically used when it is more important to process large volumes of information simultaneously, where the speed of results after receiving data is of less concern.


Installation
============================

Create a virtual environment as desired, e.g. ``python -m venv ./venv``, then:

.. code-block:: python

   cd ./molten/
   
   #for read-only:
   pip install . 

   #to allow editing, running tests, generating docs, etc.
   pip install -e .[dev] 

Testing and Documentation
============================

After installation using the ``[dev]`` option above, unit tests can be run and 
and html documentation can be generated.

Unit tests can be run with the command ``pytest``. By default, a coverage 
report with highlighting will be generated in ``htmlcov/index.html``. These
default settings are specified in ``setup.cfg`` under ``[tool:pytest]``.

HTML documentation can be generated at ``molten/docs/build/html/index.html`` with:

.. code-block:: python

   cd docs
   sphinx-apidoc -M --templatedir source/templates -f -o source ../src/molten && make clean && make html

Docker support is currently bugged. Below describes the steps you may use to build the environment with Docker and (optionally) get into the container.

.. code-block:: python

   sudo docker build -f bin/Dockerfile .        # build image
   sudo docker images                           # optionally see images
   sudo docker run -itd <image-name>            # build + start container
   sudo docker ps -a                            # optionally see containers
   sudo docker exec -it <container-name> bash   # go into container
   


