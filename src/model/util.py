import tensorflow as tf
from dataset import WindowGenerator
from typing import Dict


def compile_and_fit(
        model: tf.keras.Model,
        window: WindowGenerator,
        max_epochs: int,
        debug: bool,
        patience=5,
) -> Dict:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='mean_absolute_error',
        patience=2 * patience,
        verbose=2
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='mean_absolute_error',
        patience=patience,
        verbose=2
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError()
        ],
        run_eagerly=debug,
        jit_compile=not debug
    )

    history = model.fit(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=[
            early_stopping,
            reduce_lr
        ]
    )
    return history
