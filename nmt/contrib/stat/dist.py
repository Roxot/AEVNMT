"""
Some standard distributions.


:Authors: - Wilker Aziz
"""
import tensorflow as tf
import abc
import numpy as np
from nmt.contrib.stat.parameter import Parameter
from nmt.contrib.stat.parameter import Location
from nmt.contrib.stat.parameter import Scale


class Distribution:

    def __init__(self, param_specs: 'list[Parameter]', image: str):
        self._param_specs = tuple(param_specs)
        self._image = image

    @property
    def image(self) -> str:
        """One of real/positive/negative/prob"""
        return self._image

    @property
    def num_params(self) -> int:
        return len(self._param_specs)

    @property
    def param_specs(self) -> 'tuple[Parameter]':
        return self._param_specs

    @abc.abstractmethod
    def mean(self, params: list):
        """Compute the expected value (mean) from a list of parameters (each a tensor)"""
        pass

    @abc.abstractmethod
    def mode(self, params: list):
        """Compute the mode from a list of parameters (each a tensor)"""
        pass

    @abc.abstractmethod
    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        """Sample a tensor with shape `shape` from Dist(`params`)"""
        pass

    @abc.abstractmethod
    def kl(self, params_i: list, params_j: list):
        """Compute KL(Dist(`params_i`) || Dist(`params_j`)"""
        pass


class LocationScale(Distribution):
    """
    E ~ Dist(0, 1)
    location + scale * E ~ Dist(location, scale)
    """

    def __init__(self):
        super(LocationScale, self).__init__(param_specs=[Location('location'), Scale('scale')], image='real')

    @abc.abstractmethod
    def random_standard(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        """Sample a tensor with shape `shape` from Dist with standard parameters (location=0, scale=1)"""
        pass

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        """
        Representation:

            E ~ Dist(0, 1)
            l + s * E ~ Dist(l, s)

        :param shape:
        :param params: [location, scale]
        :param dtype:
        :param seed:
        :param name:
        :return:
        """
        location, scale = params
        epsilon = self.random_standard(shape, dtype=dtype, seed=seed, name=name)
        return location + scale * epsilon


class Normal(LocationScale):

    def __init__(self):
        super(Normal, self).__init__()

    def mean(self, params: list):
        return params[0]  # [location=mean, scale]

    def mode(self, params: list):
        return params[0]  # [location=mode, scale]

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        """
        X ~ N(0, I).

        :param shape:
        :param dtype:
        :param seed:
        :param name:
        :return:
        """
        return tf.random_normal(shape, mean=0., stddev=1., dtype=dtype, seed=seed, name=name)

    def kl(self, params_i: list, params_j: list):
        location_i, scale_i = params_i  # [mean, std]
        location_j, scale_j = params_j  # [mean, std]
        var_i = scale_i ** 2
        var_j = scale_j ** 2
        term1 = 1 / (2 * var_j) * ((location_i - location_j) ** 2 + var_i - var_j)
        term2 = tf.log(scale_j) - tf.log(scale_i)
        return term1 + term2  # tf.reduce_sum(term1 + term2, axis=-1)


class Gumbel(LocationScale):

    def __init__(self):
        super(Gumbel, self).__init__()

    def mean(self, params: list):
        location = params[0]
        return location + np.euler_gamma

    def mode(self, params: list):
        return params[0]  # [location=mode, scale]

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        u = tf.random_uniform(shape, minval=0, maxval=1., dtype=dtype, seed=seed, name=name)
        return -tf.log(-tf.log(u))

    def kl(self, params_i: list, params_j: list):
        raise ValueError('Missing an expression for KL(Gumbel_i || Gumbel_j)')


class OneParameterGumbel(Distribution):

    def __init__(self):
        super(OneParameterGumbel, self).__init__(param_specs=[Location('location')], image='real')

    def mean(self, params: list):
        location = params[0]
        return location + np.euler_gamma

    def mode(self, params: list):
        return params[0]  # [location=mode]

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        """
        Representation:

            E ~ Dist(0, 1)
            l + s * E ~ Dist(l, s)

        :param shape:
        :param params: [location, scale]
        :param dtype:
        :param seed:
        :param name:
        :return:
        """
        location = params[0]
        u = tf.random_uniform(shape, minval=0, maxval=1., dtype=dtype, seed=seed, name=name)
        epsilon = -tf.log(-tf.log(u))
        return location + epsilon

    def kl(self, params_i: list, params_j: list):
        raise ValueError('Missing an expression for KL(Gumbel(mu_i, 1) || Gumbel(mu_j, 1))')


gumbel = Gumbel()
normal = Normal()
gumbel1 = OneParameterGumbel()

