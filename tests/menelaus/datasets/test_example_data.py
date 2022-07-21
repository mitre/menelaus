from numpy import float64, int32, int64
from menelaus.datasets.make_example_data import (
    make_example_batch_data,
    fetch_circle_data,
    fetch_rainfall_data
)


def test_example_batch_data_1():
    """Ensure generated example data has correct shape"""
    df = make_example_batch_data()
    assert list(df.columns) == [
        "year",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "cat",
        "confidence",
        "drift",
    ]
    assert list(df.year.unique()) == [
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
    ]
    df_grouped = df.groupby("year")
    dfs = [x for _, x in df_grouped]
    for df in dfs:
        assert len(df) == 20000


def test_example_batch_data_2():
    """Ensure generated example data has right column types"""
    df = make_example_batch_data()
    assert df.dtypes["year"] == int64
    for char in "abcdefghij":
        assert df.dtypes[char] == float64
    assert (df.dtypes["cat"] == int32) or (df.dtypes["cat"] == int64)
    assert df.dtypes["confidence"] == float64
    assert df.dtypes["drift"] == bool


def test_circle_data():
    df = fetch_circle_data()
    assert all(df.dtypes == float64)
    assert df.shape == (2000, 3)


def test_rainfall_data():
    df = fetch_rainfall_data()
    assert all(df.dtypes == object)
    assert df.shape == (18159, 9)
