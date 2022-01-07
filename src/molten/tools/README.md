## Description of `run_experiments.py`

### Conventions

- `trial`: one specific run of `PCA_CD`
- `status`: output file describing detected drift locations for one specific `trial` run
- `configuration`: one specific calibration of all `PCA_CD` internal parameters (not including data passed in, etc.)

### File Process - `run_experiments.py`

1. A single `trial` (i.e., for one `configuration`) will output exactly one `status`, arbitraily named `<timestamp>.csv`.
2. Each unique `configuration` run will have its own 'common' directory, whose name describes the `configuration`.
3. Each `trial` with a specific `configuration`, has its `status` stored in the common directory for that `configuration`.
4. A master file will record for each `trial`: the `configuration` + results + location of the `status`. 

__Example__: your directory should look like this:

```
/experiments

    /artifacts                      # you will need this directory setup 

        master.csv                  # records timing, file locations, etc.
        fake_wls_eligibility.csv
        dataCircleGSev3Sp3Train.csv

        /p_fake_wls_kl_300_50       # PCACD, Fake WLS data, KL, WS=300, EV=50
            <timestamp>.csv         # observed drift locations for one single trial
            <timestamp>.csv         # an identical trial run at a different time
        /p_fake_wls_llh_300_50
        /p_fake_wls_llh_400_50
```

### Notes

- in `parse_experiments`, `Results.ipynb`, `Testing.ipynb`, there is some initial code to smartly read trial results
    - this code is clunky and will need revisiting
- really, experimentation is only intended for `PCA_CD` at this time
- `window_size` is, for now, also the `sample_size` (how much from each 'batch' of a data stream is put into the sample)
    - this logic is actually set in the `main` function of `run_experiments.py`
