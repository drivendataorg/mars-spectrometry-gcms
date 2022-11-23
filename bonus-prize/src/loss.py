import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def label_loss_func(y, p):
    bce = tf.keras.losses.binary_crossentropy(y, p)
    sh = tf.keras.losses.hinge(y, p)
    return bce + .1 * sh


def self_cross_corr_matrix(x):
    x = tf.cast(x, tf.float32)

    n = tf.shape(x)[0]
    d = x.shape[-1]
    dist = tfd.Normal(loc=0., scale=.5)
    x = x[:, None] + dist.sample([n, d, d])

    y = tf.transpose(x, (0, 2, 1))

    corr = tfp.stats.correlation(x, y, sample_axis=0, event_axis=None)
    return corr


def lcg_loss_func(y, p):
    y = self_cross_corr_matrix(y)
    p = self_cross_corr_matrix(p)

    loss_ = tf.abs(y - p)
    loss_ = tf.reduce_mean(loss_)
    return loss_


def loss_func(y, p):
    return label_loss_func(y, p) + 0.1 * lcg_loss_func(y, p)
