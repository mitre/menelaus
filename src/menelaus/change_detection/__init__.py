"""
Change detection algorithms monitor sequential univariate metrics. They can
either be applied to detect drift in a model’s performance metric or can be used
in the context of monitoring a single feature from a dataset. The change
detection algorithms presented in this package can detect bi-directional shifts,
either upward or downward changes in a sequence. 
"""

from menelaus.change_detection.adwin import ADWIN
from menelaus.change_detection.cusum import CUSUM
from menelaus.change_detection.page_hinkley import PageHinkley
