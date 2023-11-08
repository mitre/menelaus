"""
The experimental module is for live, but untested / informally-released code. Currently this houses
classes and functions meant to apply drift detection to NLP data. This is achieved by implementing
three components:
    * Transform functions, which are curried functions initialized with a certain configuration,
    and called in some sequence to transform an initial batch of data into a final formatted data
    representation. Applying transforms helps compare two sets of data, and convert data into a format
    accepted by some ``Alarm`` type.
    * ``Alarm`` classes, which are parameterized objects with an ``evaluate()`` function that takes 2 
    data representations (*i.e.*, transformed data) and determines whether drift has occurred according 
    to some statistical test, principle, or heuristic. 
    * A ``Detector`` class which accepts a list of transforms to apply, an initialied alarm scheme,
    and can step through newly-seen data sequentially and report drift as it occurs. Importantly, this
    class can recalibrate the source or reference data as drift is discovered.
"""
