#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import random


class sparse_autocoder(object):
    def __init__(self, layers=[], alpha=0.3, toler=0.1,
                 max_iter=10000, active_func='sigmoid'):
        self.layers = layers
        self.alpha = alpha
        self.max_iter = max_iter
        self.active_func = active_func
        self.toler = toler
        self.num_layers = len(self.layers)

    def fit(self, data_set, method='BGD'):
        self.data_mat = np.mat(data_set)
        self.weights = []
        self.bias = []
        for i in range(1, self.num_layers):
            n = self.layers[i - 1]
            m = self.layers[i]
            self.weights.append(np.random.rand(m, n) - 0.5)
            self.bias.append(np.zeros((m, 1)))

        self.gradient_descent(method)

    def gradient_descent(self, method):
        _iter = 0
        while _iter < self.max_iter:
            m, indexes = self.get_batch(method)
            actives_mean = [np.zeros((layer, 1)) for layer in self.layers]
            outs = []
            for i in indexes:
                actives = []
                target = self.data_mat[i].T
                actives.append(self.data_mat[i].T)
                for j in range(1, self.num_layers):
                    Oj = np.dot(self.weights[j - 1],
                                actives[j - 1]) + self.bias[j - 1]
                    actives.append(self.sigmoid(Oj))
                    actives_mean[j] += Oj
                outs.append(actives)
            actives_mean = [act / m for act in actives_mean]
            weights_delta = [np.zeros(w.shape) for w in self.weights]
            bias_delta = [np.zeros(b.shape) for b in self.bias]
            errors = 0
            for actives in outs:
                output_layer = self.num_layers - 1
                error = self.get_item_error(actives[output_layer], target)
                errors += error
                if error <= 0.01:
                    continue
                delta = -np.multiply(target - actives[output_layer],
                                     self.sigmoid_prime(actives[output_layer]))
                layer = output_layer - 1
                while layer >= 0:
                    weights_delta[layer] += np.dot(delta, actives[layer].T)
                    bias_delta[layer] += delta
                    delta = np.multiply(np.dot(self.weights[layer].T, delta),
                                        self.sigmoid_prime(actives[layer]))
                    layer -= 1
            for k in range(self.num_layers - 1):
                self.weights[k] -= self.alpha * (1.0 / m) * weights_delta[k]
                self.bias[k] -= self.alpha * (1.0 / m) * bias_delta[k]
            if errors / m < self.toler:
                break
            _iter += 1

    def get_batch(self, method='BGD'):
        data_size, _ = np.shape(self.data_mat)
        if method == 'BGD':  # Batch gradient descent
            indexes = range(data_size)
            m = data_size
        elif method == 'SGD':  # Stochastic gradient descent
            indexes = [random.randint(0, data_size - 1)]
            m = 1
        elif method == 'MBGD':  # Mini-batch gradient descent
            m = 10
            indexes = [random.randint(0, data_size - 1) for x in range(m)]
        return m, indexes

    def get_item_error(self, out, y):
        minus = out - y
        return np.sum(np.multiply(minus, minus)) / 2.0

    def classifier(self, x):
        outs = []
        outs.append(np.mat(x).T)
        for i in range(1, self.num_layers):
            Oi = np.dot(self.weights[i - 1],
                        outs[i - 1]) + self.bias[i - 1]
            Oi = self.sigmoid(Oi)
            outs.append(Oi)
        return outs[-1]

    def sigmoid_prime(self, y):
        return np.multiply(y, (1 - y))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
