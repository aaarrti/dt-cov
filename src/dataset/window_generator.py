import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from .constants import *
from typing import List, Optional


class WindowGenerator:
    def __init__(
            self,
            input_width: int,
            label_width: int,
            shift: int,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            label_columns: Optional[List[str]] = None
    ):
        """
        :param input_width: number of time steps NN consumes
        :param label_width: ???
        in https://www.tensorflow.org/tutorials/structured_data/time_series#1_indexes_and_offsets
        is always the same as shift
        :param shift: number of steps in the future to predict
        :param train_df: train data
        :param val_df: validation data
        :param label_columns: names of columns with features, which should be used for training

        Input width + label width must be <= totale width of the data frame
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {
            name: i for i, name in enumerate(train_df.columns)
        }

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, inputs, labels, model=None, plot_col=DEATHS_COLUMN, max_subplots=1):
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            if labels is not None:
                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data) -> tf.data.Dataset:
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self) -> tf.data.Dataset:
        return self.make_dataset(self.train_df)

    @property
    def val(self) -> tf.data.Dataset:
        return self.make_dataset(self.val_df)

    @property
    def test_inputs(self):
        ds = self.train.concatenate(self.val).unbatch()
        X = np.array([i for i, j in ds.as_numpy_iterator()])
        return X
