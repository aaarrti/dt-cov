import tensorflow as tf


class Baseline(tf.keras.Model):

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    @tf.function(jit_compile=True)
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        out = result[:, :, tf.newaxis]
        return out


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class MultiStepLastBaseline(tf.keras.Model):

    def __init__(self, out_steps):
        super().__init__()
        self.out_steps = out_steps

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.out_steps, 1])


class FeedBack(tf.keras.Model):

    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.num_features = num_features
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(self.num_features)

    @tf.function(jit_compile=True)
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


class RepeatBaseline(tf.keras.Model):

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return inputs
