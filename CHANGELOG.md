# Changelog

Notable changes to Menelaus will be documented here.

## v0.1.1 - Jun 7, 2022

- Initial public release.
- Published to pypi.
- Published to readthedocs.io.

## v0.1.2 - July 11, 2022

- Updated the documentation
- Added example jupyter notebooks to ReadTheDocs
- Switched to sphinx-bibtext for citations
- Formatting and language tweaks.
- Added StreamingDetector and BatchDetector abstract base classes.
- Re-factored kdq-tree to use new abstract base classes: the separate classes KdqTreeStreaming and KdqTreeBatch now exist.
- kdq-tree can now consume dataframes.
- Added new git workflows and improved old ones.

## v0.2.0 - December 7, 2022

- Updates to documentation.
- Updated the arguments for detector `update` and `set_reference` methods.
- Added validation to the detector `update` and `set_reference` methods.
- Added the `datasets` module, which contains or generates example datasets.
- Added implementation of Margin Density Drift Detection (MD3) semi-supervised detector.
- Added implementation of Nearest Neighbor-based Density Variation Identification (NN-DVI) data drift detector.
- Added Ensemble wrapper that allows combining two or more drift detectors.