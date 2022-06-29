from .models import *
from dataset import load_df, DEATHS_COLUMN, OUT_STEPS, CONV_WIDTH


'''
Models for prediction single time series
'''


def base_line_model() -> tf.keras.Model:
    df = load_df()
    columns = df.columns
    column_indices = {name: i for i, name in enumerate(columns)}
    nn = Baseline(column_indices[DEATHS_COLUMN])
    return nn


def linear_model() -> tf.keras.Model:
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
    return linear


def dense_model() -> tf.keras.Model:
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    return dense


def conv_model(conv_width=3) -> tf.keras.Model:
    nn = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    return nn


def rnn_model() -> tf.keras.Model:
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return lstm_model


def residual_rnn_model() -> tf.keras.Model:
    df = load_df()
    num_features = df.shape[1]
    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                num_features,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros()
            )
        ])
    )
    return residual_lstm


'''
Models for predicting multiple time series
'''


def multi_step_baseline_model() -> tf.keras.Model:
    return MultiStepLastBaseline(out_steps=OUT_STEPS)


def repeat_baseline_model() -> tf.keras.Model:
    # FIXME
    return RepeatBaseline()


def multi_linear_model() -> tf.keras.Model:
    df = load_df()
    num_features = df.shape[1]
    nn = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return nn


def multi_dense_model() -> tf.keras.Model:
    df = load_df()
    num_features = df.shape[1]
    nn = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation='relu'),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return nn


def multi_conv_model() -> tf.keras.Model:
    df = load_df()
    num_features = df.shape[1]
    nn = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return nn


def multi_lstm_model() -> tf.keras.Model:
    df = load_df()
    num_features = df.shape[1]
    nn = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return nn


def auto_regressive_model() -> tf.keras.Model:
    df = load_df()
    num_features = df.shape[1]
    return FeedBack(
        units=24,
        num_features=num_features,
        out_steps=OUT_STEPS
    )
