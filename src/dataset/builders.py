from .window_generator import WindowGenerator
from .util import load_split_df
from .constants import *


def single_step_window() -> WindowGenerator:
    train_df, val_df = load_split_df()
    wg = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        label_columns=[DEATHS_COLUMN],
        train_df=train_df,
        val_df=val_df
    )
    return wg


def wide_single_step_window() -> WindowGenerator:
    train_df, val_df = load_split_df()
    width = min(train_df.shape[0] * train_df.shape[1], val_df.shape[0] * val_df.shape[1])
    wg = WindowGenerator(
        input_width=width,
        label_width=width,
        shift=1,
        label_columns=[DEATHS_COLUMN],
        train_df=train_df,
        val_df=val_df
    )

    return wg


def conv_single_step_window() -> WindowGenerator:
    train_df, val_df = load_split_df()
    wg = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=[DEATHS_COLUMN],
        train_df=train_df,
        val_df=val_df
    )

    return wg


def wide_conv_single_step_window() -> WindowGenerator:
    train_df, val_df = load_split_df()
    width = min(train_df.shape[0] * train_df.shape[1], val_df.shape[0] * val_df.shape[1])

    input_width = width + (CONV_WIDTH - 1)

    wg = WindowGenerator(
        input_width=input_width,
        label_width=width,
        shift=1,
        label_columns=[DEATHS_COLUMN],
        train_df=train_df,
        val_df=val_df
    )
    return wg


def multi_window() -> WindowGenerator:
    train_df, val_df = load_split_df()
    width = min(train_df.shape[0] * train_df.shape[1], val_df.shape[0] * val_df.shape[1])
    wg = WindowGenerator(
        input_width=width,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        train_df=train_df,
        val_df=val_df
    )
    return wg
