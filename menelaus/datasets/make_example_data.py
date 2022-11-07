""" Functions to generate example data according to a fixed scheme. """

import os
import pandas as pd
import numpy as np


def make_example_batch_data():
    """
    This function returns a dataframe containing synthetic batch data for use
    with the repo's examples. The dataframe's columns are ``"year", "a", "b", ... "j", "cat", "confidence", "drift"``.

        * ``year`` covers 2007-2021, with 20,000 observations each.

        * Features ``"b", "e", "f"`` are normally distributed.

        * Features ``"a", "c", "d", "g", "h", "i", "j"`` have a gamma distribution.

        * The ``"cat"`` feature contains categorical variables ranging from 1-7,
          sampled with varying probability.

        * ``"confidence"`` contains values on ``[0, 0.6]`` through 2018, then values
          on ``[0.4, 1]``.

    Drift occurs as follows:

        * Change the mean of column ``"b"`` in 2009. Reverts to original distribution
          in 2010.

        * Change the variance of columns ``"c"`` and ``"d"`` in 2012 by replacing
          some samples with the mean. Reverts to original distribution in 2013.

        * Increase the correlation of columns ``"e"`` and ``"f"`` in 2015 (0 correlation
          to 0.5 correlation).

        * Change the mean and variance of column ``"h"`` in 2019, and maintain this
          new distribution going forward. Change the range of the "confidence"
          column going forward.

        * Change the mean and variance of column ``"j"`` in 2021.

    Returns:
        pd.DataFrame: A dataframe containing a synthetic batch dataset.
    """
    np.random.seed(123)
    year_size = 20000
    df = pd.DataFrame()
    df["year"] = year_size * list(range(2007, 2022))
    df.sort_values(by="year", inplace=True)
    df.reset_index(inplace=True)
    sample_size = df.shape[0]

    df["a"] = np.random.gamma(shape=8, size=sample_size) * 1000
    df["b"] = np.random.normal(loc=200, scale=10, size=sample_size)
    df["c"] = np.random.gamma(shape=7, size=sample_size) * 1000
    df["d"] = np.random.gamma(shape=10, size=sample_size) * 10000
    df[["e", "f"]] = np.random.multivariate_normal(
        mean=(0, 0), cov=np.array([[2, 0], [0, 2]]), size=sample_size
    )
    df["g"] = np.random.gamma(shape=11, size=sample_size) * 10000
    df["h"] = np.random.gamma(shape=12, size=sample_size) * 1000
    df["i"] = np.random.gamma(shape=9, size=sample_size) * 1000
    df["j"] = np.random.gamma(shape=10, size=sample_size) * 100
    df["cat"] = np.random.choice(
        range(7), size=sample_size, p=(0.3, 0.3, 0.2, 0.1, 0.05, 0.04, 0.01)
    )
    df["confidence"] = np.random.uniform(low=0, high=0.6, size=sample_size)

    ######################################################################
    # Drift 1: change the mean of B in 2009, means will revert for 2010 on
    df.loc[df.year == 2009, "b"] = np.random.normal(size=year_size, loc=500, scale=10)

    ######################################################################
    # Drift 2: change the variance of c and d in 2012 by replacing some with the mean
    # keep same mean as other years, revert by 2013
    mu_c = df["c"].mean()
    mu_d = df["d"].mean()

    # subtle change, every 10 obs
    df.loc[(df.year == 2012) & (df.index % 10 == 0), "c"] = mu_c + np.random.normal(
        loc=0, scale=10, size=year_size // 10
    )

    # bigger change, every other obs
    df.loc[(df.year == 2012) & (df.index % 2 == 0), "d"] = mu_d + np.random.normal(
        loc=0, scale=10, size=year_size // 2
    )

    ######################################################################
    # Drift 3: change the correlation of e and f in 2015 (go from correlation of 0 to correlation of 0.5)
    df.loc[df.year == 2015, ["e", "f"]] = np.random.multivariate_normal(
        mean=(0, 0), cov=np.array([[2, 1], [1, 2]]), size=year_size
    )

    ######################################################################
    # Drift 4: change mean and var of H and persist it from 2018 on, change range of confidence scores
    df.loc[df.year > 2018, "h"] = (
        np.random.gamma(shape=1, scale=1, size=3 * year_size) * 1000
    )
    df.loc[df.year > 2018, "confidence"] = np.random.uniform(
        low=0.4, high=1, size=3 * year_size
    )

    ######################################################################
    # Drift 5: change mean and var just for a year of J in 2021
    df.loc[df.year == 2021, "j"] = np.random.gamma(shape=10, size=year_size) * 10

    df["drift"] = df["year"].isin([2009, 2012, 2015, 2018, 2021])
    df.drop("index", axis=1, inplace=True)
    return df


def fetch_circle_data():
    """Retrieve the Circle data from the datasets directory. Circle is synthetic
    data containing drift due to both a change in the feature distribution and a
    change in the conditional target distribution. Drift occurs from index
    1000-1250 and affects 66% of the data points.

    Ref. :cite:t:`minku2010`

    Returns:
        pd.DataFrame: A dataframe containing the Circle dataset.
    """
    data_path = os.path.join(os.path.dirname(__file__), "dataCircleGSev3Sp3Train.csv")
    return pd.read_csv(data_path, usecols=[0, 1, 2], names=["var1", "var2", "y"])


def fetch_rainfall_data():
    """Retrieve the Rainfall data from the datasets directory. National Oceanic
    and Atmospheric Administration (NOAA) rainfall data contains weather
    measurements collected over a 50 year period at a site location in
    Bellevue, Nebraska. It contains eight features: temperature, dew point,
    sea-level pressure, visibility, average wind speed, max sustained wind-speed,
    minimum temperature, and maximum temperature. The dependent variable is rain.
    Concept and data drift starts in index 12,000 and persists through the rest
    of the dataset.

    Ref. :cite:t:`souza2020`

    Returns:
        pd.DataFrame: A dataframe containing the Rainfall dataset.

    """
    data_path = os.path.join(os.path.dirname(__file__), "rainfall_data.csv")
    df = pd.read_csv(
        data_path,
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        names=[
            "index",
            "temperature",
            "dew_point",
            "sea_level_pressure",
            "visibility",
            "average_wind_speed",
            "max_sustained_wind_speed",
            "minimum_temperature",
            "maximum_temperature",
            "rain",
        ],
    )
    df = df.iloc[1:, :].reset_index(drop=True)
    df = df.apply(pd.to_numeric)
    return df
