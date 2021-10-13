# molten

- for now, pip install -e . will resolve the references, before running examples/Testing Vignette.ipynb
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