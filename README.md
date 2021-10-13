# Notes on use
- need to make some decisions about setup.py, probably
- packages:
    - jupyter matplotlib pandas numpy sklearn statistics plotly scipy
    - jupyter isn't a strict requirements for the detectors, but is necessary for Testing_Vignette2.ipynb
    - Separating environments with conda, but installing with pip to avoid conda nonsense, works:
        ```
        cd ./molten/
        conda install pip
        pip install -r requirements.txt
        ```
# Background

Concept drift is an established phenomenon in machine learning (ML) and predictive analytics in which the performance of a model changes over time. There is very little published work on effectively integrating drift detection in the clinical space. However, standards of care, disease prevalence, and target population characteristics are rarely static over time. After an algorithm has been implemented, how do we know if the outcomes or features will change over time in such a way that degrades model performance? Perhaps even rendering the model dangerous to patients or leading to gross overutilization? The MOLTEN (MOdel Longevity Test ENgine) team synthesized drift detection and mitigation best practices for a clinical audience, e.g. electronic health records-based (EHR) datasets, as well as applied nearly a dozen drift detectors to two real-world EHR datasets. 

FY21 MIP, MOLTEN: https://mitre.spigit.com/mipfy21/Page/ViewIdea?ideaid=109457

FY22 MIP, iMOLTEN: https://mitre.spigit.com/mipfy22/Page/ViewIdea?ideaid=115313
