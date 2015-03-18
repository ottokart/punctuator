# coding: utf-8

import numpy as np

# y - activation of the unit given input z
# dy_dz - derivative of the activation y of the unit with respect to the input z

class Sigmoid(object):

    @staticmethod
    def y(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dy_dz(y):
        return y * (1. - y)

class Softmax(object):

    @staticmethod
    def y(z):
        # Subtract maximum for numerical stability. (does not affect outputs)
        y = np.exp(z - z.max(axis=1)[:,np.newaxis])
        y /= y.sum(axis=1)[:,np.newaxis]
        return y

class Linear(object):

    @staticmethod
    def y(z):
        return z

    @staticmethod
    def dy_dz(y):
        return 1.

class RectifiedLinear(object):

    @staticmethod
    def y(z):
        return np.maximum(z, 0.)

    @staticmethod
    def dy_dz(y):
        return y > 0.

class Tanh(object):

    @staticmethod
    def y(z):
        return np.tanh(z)

    @staticmethod
    def dy_dz(y):
        return 1. - y**2
