"""
Helper functions for standard pdfs and pmfs.

:Authors: - Wilker Aziz
"""
import tensorflow as tf


def beta_fn(a, b):
    # A useful identity:
    #   B(a,b) = exp(log Gamma(a) + log Gamma(b) - log Gamma(a+b))
    # but here we simply exponentiate tf.lbeta instead, feel free to use whichever version you prefer
    return tf.exp(tf.lbeta(tf.concat([tf.expand_dims(a, -1), tf.expand_dims(b, -1)], -1)))

