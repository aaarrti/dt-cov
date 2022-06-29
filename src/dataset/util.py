from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import dateutil.parser as dp

from .constants import *
from functools import lru_cache


def split_dataset(dataset: pd.DataFrame, test_ratio=0.2) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a panda dataframe in two.
    """
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


@lru_cache(maxsize=None)
def load_df() -> pd.DataFrame:
    df = pd.read_csv('https://raw.githubusercontent.com/jgehrcke/covid-19-germany-gae/master/deaths-rki-by-ags.csv')
    X = list(df[TIME_COLUMN])
    unix_time = list(map(lambda ts: int(dp.parse(ts).timestamp()), X))
    df[UNIX_TIME_COLUMN] = unix_time

    df = df[[UNIX_TIME_COLUMN, DEATHS_COLUMN]]
    return df


@lru_cache(maxsize=None)
def load_split_df() -> (pd.DataFrame, pd.DataFrame):
    df = load_df()
    train, val = split_dataset(df)
    return train, val



