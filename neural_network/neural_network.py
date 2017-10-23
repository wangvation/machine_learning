#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Neural_Network(object):
    """docstring for Neural_Network"""

    def __init__(self, layers=[], step=0.3,
                 max_iter=10000, active_func='sigmoid'):
        self.layers = layers
        self.step = step
        self.max_iter = max_iter
        self.active_func = active_func

    def fit(self, data_set, label_set):
        self.data_mat = np.mat(data_set)
        self.label_set = np.mat(label_set)
        self.weights = []
        self.bias = []
        layer_size = len(self.layers)
        for i in range(1, layer_size):
            n = self.layers[i - 1]
            m = self.layers[i]
            self.weights.append(np.random.rand(m, n))
            self.bias.append(np.random.rand(m, 1))

        self.gradient_descent()

    def gradient_descent(self):
        layer_size = len(self.layers)
        _iter = 0
        while _iter < self.max_iter:
            outs = []
            targets = self.label_set[0]
            outs.append(self.data_mat[0])
            for i in range(1, layer_size):
                Oi = np.dot(self.weights[i - 1],
                            outs[i - 1].T) + self.bias[i - 1]
                Oi = self.simmoid(Oi)
                outs.append(Oi)
            i = layer_size - 1
            while i >= 0:
                delta = np.multiply(outs[i], (1 - outs[i]))
                delta = np.multiply(delta, outs[i] - targets)
                for j in range(self.layers[i - 1]):
                    self.weights[i][:, j] -= self.step * delta * outs[i][j]
                self.bias[i] -= self.step * delta
                i -= 1
                pass
            _iter += 1

        pass

    def simmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
