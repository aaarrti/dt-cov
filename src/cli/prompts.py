from PyInquirer import style_from_dict, Token, prompt
from typing import List

from dataset import *
from model import *

style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

_SINGLE = 'single'
_MULTI = 'multi'
_WINDOW = 'window'
_TYPE = 'type'
_LIST = 'list'
_MODEL = 'model'

AVAILABLE_PREDICTION_TYPES = [_SINGLE, _MULTI]

_BASELINE_MODEL = 'base_line_model'
_LINEAR_MODEL = 'linear_model'
_DENSE_MODEL = 'dense_model'
_CONV_MODEL = 'conv_model'
_RNN_MODEL = 'rnn_model'
_RESIDUAL_RNN_MODEL = 'residual_rnn_model'

_MULTI_BASELINE_MODEL = 'multi_step_baseline_model'
_REPEAT_BASELINE_MODEL = 'repeat_baseline_model'
_MULTI_LINEAR_MODEL = 'multi_linear_model'
_MULTI_DENSE_MODEL = 'multi_dense_model'
_MULTI_CONV_MODEL = 'multi_conv_model'
_MULTI_LSTM_MODEL = 'multi_lstm_model'
_AUTO_REGRESSIVE_MODEL = 'auto_regressive_model'

AVAILABLE_SINGLE_STEP_MODELS = [
    _BASELINE_MODEL,
    _LINEAR_MODEL,
    _DENSE_MODEL,
    _CONV_MODEL,
    _RNN_MODEL,
    _RESIDUAL_RNN_MODEL
]

AVAILABLE_MULTI_STEP_MODELS = [
    _MULTI_BASELINE_MODEL,
    _REPEAT_BASELINE_MODEL,
    _MULTI_LINEAR_MODEL,
    _MULTI_DENSE_MODEL,
    _MULTI_CONV_MODEL,
    _MULTI_LSTM_MODEL,
    _AUTO_REGRESSIVE_MODEL
]

MODEL_BUILDERS = {
    _BASELINE_MODEL: base_line_model,
    _LINEAR_MODEL: linear_model,
    _DENSE_MODEL: dense_model,
    _CONV_MODEL: conv_model,
    _RNN_MODEL: rnn_model,
    _RESIDUAL_RNN_MODEL: residual_rnn_model,

    _MULTI_BASELINE_MODEL: multi_step_baseline_model,
    _REPEAT_BASELINE_MODEL: repeat_baseline_model,
    _MULTI_LINEAR_MODEL: multi_linear_model,
    _MULTI_DENSE_MODEL: multi_dense_model,
    _MULTI_CONV_MODEL: multi_conv_model,
    _MULTI_LSTM_MODEL: multi_lstm_model,
    _AUTO_REGRESSIVE_MODEL: auto_regressive_model

}

_SINGLE_STEP_WINDOW = 'single_step_window'
_WIDE_WINDOW = 'wide_window'
_CONV_SINGLE_STEP_WINDOW = 'conv_single_step_window'
_CONV_WIDE_SINGLE_STEP_WINDOW = 'conv_wide_single_step_window'

_MULTI_WINDOW = 'multi_window'

AVAILABLE_WINDOWS = {
    _BASELINE_MODEL: [
        _SINGLE_STEP_WINDOW,
        _WIDE_WINDOW
    ],
    _LINEAR_MODEL: [
        _SINGLE_STEP_WINDOW,
        _WIDE_WINDOW
    ],
    _DENSE_MODEL: [
        _SINGLE_STEP_WINDOW,
        _WIDE_WINDOW
    ],
    _CONV_MODEL: [
        _CONV_SINGLE_STEP_WINDOW,
        _CONV_WIDE_SINGLE_STEP_WINDOW
    ],
    _RNN_MODEL: [
        _WIDE_WINDOW
    ],
    _RESIDUAL_RNN_MODEL: [
        _WIDE_WINDOW
    ],

    _MULTI_BASELINE_MODEL: [
        _MULTI_WINDOW
    ],
    _REPEAT_BASELINE_MODEL: [
        _MULTI_WINDOW
    ],
    _MULTI_LINEAR_MODEL: [
        _MULTI_WINDOW
    ],
    _MULTI_DENSE_MODEL: [
        _MULTI_WINDOW
    ],
    _MULTI_CONV_MODEL: [
        _MULTI_WINDOW
    ],
    _MULTI_LSTM_MODEL: [
        _MULTI_WINDOW
    ],
    _AUTO_REGRESSIVE_MODEL: [
        _MULTI_WINDOW
    ]
}

WINDOW_BUILDERS = {
    _SINGLE_STEP_WINDOW: single_step_window,
    _WIDE_WINDOW: wide_single_step_window,
    _CONV_SINGLE_STEP_WINDOW: conv_single_step_window,
    _CONV_WIDE_SINGLE_STEP_WINDOW: wide_conv_single_step_window,

    _MULTI_WINDOW: multi_window,
}


def _prompt(options: List[str]) -> (tf.keras.Model, WindowGenerator):
    model_question = [
        {
            'type': _LIST,
            'message': 'Select model',
            'name': _MODEL,
            'choices': [{'name': i, 'value': i} for i in options]
        }
    ]
    model_answer = prompt(model_question, style=style)
    model_name = model_answer[_MODEL]
    nn: tf.keras.Model = MODEL_BUILDERS[model_name]()

    window_question = [
        {
            'type': _LIST,
            'message': 'Select window',
            'name': _WINDOW,
            'choices': [{'name': i, 'value': i} for i in AVAILABLE_WINDOWS[model_name]]
        }
    ]

    window_answer = prompt(window_question, style=style)
    window_name = window_answer[_WINDOW]

    wg: WindowGenerator = WINDOW_BUILDERS[window_name]()

    return nn, wg


def prompt_model_and_window() -> (WindowGenerator, tf.keras.Model):
    prediction_question = [
        {
            'type': _LIST,
            'message': 'Select prediction type',
            'name': _TYPE,
            'choices': [{'name': i, 'value': i} for i in AVAILABLE_PREDICTION_TYPES],
            'default': _SINGLE
        }
    ]
    prediction_answer = prompt(prediction_question, style=style)
    if prediction_answer[_TYPE] == _SINGLE:
        return _prompt(AVAILABLE_SINGLE_STEP_MODELS)
    else:
        return _prompt(AVAILABLE_MULTI_STEP_MODELS)
