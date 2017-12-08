#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class fully_connect_layer(object):
    """
    fully connect layer
    active:active function
    active_derive:active derive
    layers:layer list
    input_array:input array
    alpha:learning rate
    """

    def __init__(self, active, active_derive, layers):
        '''
        Parameters
        ----------
        active:active function
        active_derive:active derive
        layers:layer list
        input_array:input array

        '''
        self.active = active
        self.active_derive = active_derive
        std = 1.0 / np.sqrt(layers[1] * layers[0])
        weights = np.random.normal(loc=0.0,
                                   scale=std,
                                   size=(layers[1], layers[0]))
        self.weights = np.mat(weights)
        self.bias = np.zeros((layers[1], 1))
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)

    def forward(self, input_array):
        '''
        Parameters
        ----------
        active:active function
        active_derive:active derive
        layers:layer list
        input_array:input array
        alpha:learning rate

        '''
        self.input_mat = np.mat(input_array)
        weighted_sum = self.weights * self.input_mat + self.bias
        self.out_put = self.active(weighted_sum)
        return self.out_put

    def backward(self, delta_mat):
        self.delta_mat = np.multiply(self.weights.T * delta_mat,
                                     self.active_derive(self.out_put))
        self.clac_gradient(delta_mat)
        return self.delta_mat

    def clac_gradient(self, delta_mat):
        self.weights_grad += delta_mat * self.input_mat.T
        self.bias_grad += delta_mat

    def update(self, alpha, batch_size):
        """

        Args:
          batch_size:batch size
          learning_rate: learning rate
        Returns:

        """
        self.weights -= self.alpha * self.weights_grad / batch_size
        self.bias -= self.alpha * self.bias_grad / batch_size
