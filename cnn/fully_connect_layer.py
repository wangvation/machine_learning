#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class fully_connect_layer(object):
    """
    fully connect layer
    active:active function
    active_derive:active derive
    cost_function:cost function
    layers:layer list
    input_array:input array
    target_array:target array
    alpha:learning rate
    """

    def __init__(self, active, active_derive, cost_function,
                 layers, input_array, target_array, alpha=0.3):
        '''
        Parameters
        ----------
        active:active function
        active_derive:active derive
        cost_function:cost function
        layers:layer list
        input_array:input array
        target_array:target array
        alpha:learning_rate

        '''
        self.layers = layers
        self.alpha = alpha
        self.active = active
        self.active_derive = active_derive
        self.cost_function = cost_function
        self.num_layers = len(layers)
        self.input_mat = np.mat(input_array)
        self.target_mat = np.mat(target_array)
        self.sensitivity_map = None
        self.input_shape = input_array.shape

    def fit(self, ):
        self.weights = []
        self.bias = []
        layer_size = len(self.layers)
        for i in range(1, layer_size):
            n = self.layers[i - 1]
            m = self.layers[i]
            self.weights.append(np.random.rand(m, n) - 0.5)
            self.bias.append(np.random.rand(m, 1) - 0.5)

        self.train()

    def forward(self):
        outs = []
        outs.append(self.input_mat)
        for layer in range(1, self.num_layers):
            Oj = np.dot(self.weights[layer - 1],
                        outs[layer - 1]) + self.bias[layer - 1]
            outs.append(self.active(Oj))
        return outs

    def back_progation(self, weights_delta, bias_delta, out):
        output_layer = self.num_layers - 1
        delta = -np.multiply(self.target_mat - out[output_layer],
                             self.active_derive(out[output_layer]))
        layer = output_layer - 1
        while layer >= 0:
            weights_delta[layer] += np.dot(delta, out[layer].T)
            bias_delta[layer] += delta
            delta = np.multiply(np.dot(self.weights[layer].T, delta),
                                self.active_derive(out[layer]))
            layer -= 1
        self.sensitivity_map = delta.reshape(self.input_shape)
        return weights_delta, bias_delta

    def get_sensitivity_map(self):
        return self.sensitivity_map

    def train(self):
        weights_delta = [np.zeros(w.shape) for w in self.weights]
        bias_delta = [np.zeros(b.shape) for b in self.bias]
        outs = self.forward()
        weights_delta, bias_delta = self.back_progation(
            weights_delta, bias_delta, outs)
        for k in range(self.num_layers - 1):
            self.weights[k] -= self.alpha * weights_delta[k]
            self.bias[k] -= self.alpha * bias_delta[k]

    def out_put(self, x):
        outs = self.forward(x)
        return outs[-1]
