import tensorflow as tf
import click

from cli import prompt_model_and_window
from model import compile_and_fit


@click.command()
@click.option('--debug', default=False)
@click.option('--epochs', default=5)
def main(
        debug: bool,
        epochs: int
):
    if debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    nn, wg = prompt_model_and_window()
    print(wg)

    test_X = wg.test_inputs

    compile_and_fit(
        model=nn,
        window=wg,
        max_epochs=epochs,
        debug=debug
    )

    wg.plot(
        inputs=test_X,
        labels=None,
        model=nn
    )


if __name__ == '__main__':
    main()
