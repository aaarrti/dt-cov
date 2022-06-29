import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from datetime import datetime


# @tf.function
def time_to_unix(x, y):
    t = datetime.fromisoformat(x).second
    return x, y


def main():

    df = pd.read_csv('cases-rki.csv')

    X = df['time-iso8601']
    Y = df['DE-BB']

    ds = tf.data.Dataset.from_tensor_slices(
        (X, Y)
    )

    ds = ds.map(time_to_unix).cache().shuffle(100).prefetch(tf.data.AUTOTUNE)

    model = tfdf.keras.GradientBoostedTreesModel(task = tfdf.tasks.Regression)
    model.fit(x=ds)


if __name__ == '__main__':
    main()

