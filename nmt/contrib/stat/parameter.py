"""
Code to specify constraints on parameters of distributions.

:Authors: - Wilker Aziz
"""
import tensorflow as tf


class Parameter:
    """
    This class helps predict parameters by setting an appropriate activation to convert from the real line
    to some subset of it.
    """

    def __init__(self, activation_fn, name: str):
        self.activation_fn = activation_fn
        self.name = name


class Location(Parameter):
    """
    Location parameters live in the real line.
    """

    def __init__(self, name: str):
        super(Location, self).__init__(tf.identity, name)


class Scale(Parameter):
    """
    Scale parameters live in the positive real line.
    """

    def __init__(self, name: str):
        super(Scale, self).__init__(tf.nn.softplus, name)


class Rate(Parameter):
    """
    Rate parameters live in the positive real line.
    """

    def __init__(self, name: str):
        super(Rate, self).__init__(tf.nn.softplus, name)


class Shape(Parameter):
    """
    Shape parameters live in the positive real line.
    """

    def __init__(self, name: str):
        super(Shape, self).__init__(tf.nn.softplus, name)


class Probability(Parameter):
    """
    Probability parameters live in the interval [0, 1]
    """

    def __init__(self, name: str):
        super(Probability, self).__init__(tf.sigmoid, name)
