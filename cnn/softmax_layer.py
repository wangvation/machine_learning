#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from cnn_utils import *


class softmax_layer(object):
    """docstring for softmax_layer"""

    def __init__(self, action, action_derive, layers):
        self.layers = layers
        self.input_shape = (layers[0], 1)
        # std = np.sqrt(6.0 / layers[1] + layers[0])
        self.weights = np.random.normal(loc=0.0,
                                        scale=0.03,
                                        size=(layers[1], layers[0]))
        self.bias = np.zeros((layers[1], 1))
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        self.action_derive = action_derive

    def forward(self, input):
        self.input = input
        debug(True, 'softmax_layer input:', np.max(self.input),
              np.min(self.input), np.mean(self.input))
        self.weighted_sum = np.dot(self.weights, self.input) + self.bias
        self.out_put = self.softmax(self.weighted_sum)
        return self.out_put

    def softmax(self, x):
        max_x = np.max(x)
        epx_x = np.exp(x - max_x)
        _sum = np.sum(epx_x)
        return epx_x / _sum

    def get_error(self, target):
        return self.cross_entropy(self.weighted_sum, target)

    def cross_entropy(self, z, target):
        j = np.argmax(target)
        _max = np.max(z)
        return -np.log(np.exp(z[j] - _max) / np.sum(np.exp(z - _max)))

    def backward(self, targets):
        delta_map = self.out_put - targets
        debug(True, 'softmaxlayer-backward:', self.layers,
              np.max(delta_map), np.min(delta_map), np.mean(delta_map))
        self.delta_map = np.multiply(np.dot(self.weights.T, delta_map),
                                     self.action_derive(self.input))
        self.clac_gradient(delta_map)
        return self.delta_map

    def clac_gradient(self, delta_map):
        # print('fully--clac_gradient:', self.input_shape, np.sum(delta_map))
        self.weights_grad += np.dot(delta_map, self.input.T)
        self.bias_grad += delta_map

    def update(self, alpha, batch_size):
        """

        Args:
          batch_size:batch size
          learning_rate: learning rate
        Returns:

        """
        # print('softmax--:', self.input_shape,
        #       np.sum(self.weights_grad), alpha, batch_size)
        self.weights -= alpha * self.weights_grad / batch_size
        self.bias -= alpha * self.bias_grad / batch_size
        self.weights_grad[...] = 0.0
        self.bias_grad[...] = 0.0
