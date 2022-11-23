import tensorflow as tf
import numpy as np
from src.loss import loss_func


MEAN = np.array([[-2.224, -0.719, -1.798, -2.005, -3.783, -3.223, -3.405, -3.625, -1.777]])


def encoder_three_layer(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(2 * dff),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(dff),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(d_model)
    ])


def get_model(learning_rate=1e-5):
    dnn = encoder_three_layer(9, 512)

    x = tf.keras.Input((2268,))
    logits = dnn(x) + MEAN
    out = tf.keras.activations.sigmoid(logits)

    model = tf.keras.Model(x, out)

    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=opt, loss=loss_func,
                  metrics=[tf.keras.metrics.binary_crossentropy])

    return model

