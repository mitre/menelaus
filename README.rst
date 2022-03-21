|pipeline|

.. |pipeline| image:: https://gitlab.mitre.org/lnicholl/molten/badges/dev/pipeline.svg
   :target: https://gitlab.mitre.org/lnicholl/molten/-/commits/dev

|coverage|

.. |coverage| image:: https://gitlab.mitre.org/lnicholl/molten/badges/dev/coverage.svg
   :target: https://gitlab.mitre.org/lnicholl/molten/-/commits/dev

Installation from local repo
=========

Create a virtual environment as desired, e.g. ``python -m venv ./venv``, then:

.. code-block:: python

   cd ./molten/
   pip install . #for read-only
   pip install -e .[dev] #to allow editing, running tests, generating docs, etc.

Docker support is currently bugged. Below describe the steps you may use to build the environment with Docker and (optionally) get into the container.

.. code-block:: python

   sudo docker build -f bin/Dockerfile .        # build image
   sudo docker images                           # optionally see images
   sudo docker run -itd <image-name>            # build + start container
   sudo docker ps -a                            # optionally see containers
   sudo docker exec -it <container-name> bash   # go into container
   



- HTML documentation generation, if installing via molten[dev] as above:

.. code-block:: python

   cd docs
   sphinx-apidoc -M -f -o source ../src/molten #-f may not always be necessary.
   make html


Background
==========

Concept drift is an established phenomenon in machine learning (ML) and
predictive analytics in which the performance of a model changes over
time. There is very little published work on effectively integrating
drift detection in the clinical space. However, standards of care,
disease prevalence, and target population characteristics are rarely
static over time. After an algorithm has been implemented, how do we
know if the outcomes or features will change over time in such a way
that degrades model performance? Perhaps even rendering the model
dangerous to patients or leading to gross overutilization? The MOLTEN
(MOdel Longevity Test ENgine) team synthesized drift detection and
mitigation best practices for a clinical audience, e.g. electronic
health records-based (EHR) datasets, as well as applied nearly a dozen
drift detectors to two real-world EHR datasets.

FY21 MIP, MOLTEN:
https://mitre.spigit.com/mipfy21/Page/ViewIdea?ideaid=109457

FY22 MIP, iMOLTEN:
https://mitre.spigit.com/mipfy22/Page/ViewIdea?ideaid=115313
