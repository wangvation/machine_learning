#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import math


class neural_network(object):
    """docstring for neural_network"""

    def __init__(self, layers=[], alpha=0.3, toler=0.1,
                 max_iter=10000, active_func='sigmoid'):
        self.layers = layers
        self.alpha = alpha
        self.max_iter = max_iter
        self.active_func = active_func
        self.toler = toler

    def fit(self, data_set, label_set, method='BGD', cost='MSE'):
        self.data_mat = np.mat(data_set)
        self.label_mat = np.mat(label_set)
        self.weights = []
        self.bias = []
        layer_size = len(self.layers)
        for i in range(1, layer_size):
            n = self.layers[i - 1]
            m = self.layers[i]
            self.weights.append(np.random.rand(m, n) - 0.5)
            self.bias.append(np.random.rand(m, 1) - 0.5)

        self.gradient_descent(method, cost)

    def forward(self, input, weights, bias):
        outs = []
        outs.append(input.T)
        layer_size = len(self.layers)
        for j in range(1, layer_size):
            Oj = np.dot(weights[j - 1],
                        outs[j - 1]) + bias[j - 1]
            outs.append(self.sigmoid(Oj))
        return outs

    def cross_entropy(self, a, y):
        return -(y * math.log(a) + (1 - y) * math.log(1 - a))

    def softmax(self, x):
        _sum = np.sum(x)
        return x / _sum

    def relu(self, x):
        return np.apply_along_axis(func1d=lambda item: 0 if item < 0 else item,
                                   axis=1, arr=x)

    def derived_relu(self, y):
        return np.apply_along_axis(func1d=lambda item: 0 if item <= 0 else 1,
                                   axis=1, arr=y)

    def gradient_descent(self, method, cost):
        layer_size = len(self.layers)
        data_size, _ = np.shape(self.data_mat)
        _iter = 0
        while _iter < self.max_iter:
            if method == 'BGD':  # Batch gradient descent
                indexes = range(data_size)
                m = data_size
            elif method == 'SGD':  # Stochastic gradient descent
                indexes = [random.randint(0, data_size - 1)]
                m = 1
            elif method == 'MBGD':  # Mini-batch gradient descent
                m = 10
                indexes = [random.randint(0, data_size - 1) for x in range(m)]
            weights_delta = [np.zeros(w.shape) for w in self.weights]
            bias_delta = [np.zeros(b.shape) for b in self.bias]
            errors = 0
            for i in indexes:
                outs = []
                target = self.label_mat[i].T
                outs.append(self.data_mat[i].T)
                for j in range(1, layer_size):
                    Oj = np.dot(self.weights[j - 1],
                                outs[j - 1]) + self.bias[j - 1]
                    outs.append(self.sigmoid(Oj))
                output_layer = layer_size - 1
                error = self.get_item_error(outs[output_layer], target)
                errors += error
                if error <= 0.01:
                    continue
                if cost == 'cross_entropy' or cost == 'log_likelihood':
                    delta = outs[output_layer] - target
                else:
                    delta = -np.multiply(target - outs[output_layer],
                                         self.sigmoid_prime(outs[output_layer]))
                layer = output_layer - 1
                while layer >= 0:
                    weights_delta[layer] += np.dot(delta, outs[layer].T)
                    bias_delta[layer] += delta
                    delta = np.multiply(np.dot(self.weights[layer].T, delta),
                                        self.sigmoid_prime(outs[layer]))
                    layer -= 1
            for k in range(layer_size - 1):
                self.weights[k] -= self.alpha * (1.0 / m) * weights_delta[k]
                self.bias[k] -= self.alpha * (1.0 / m) * bias_delta[k]
            if errors / m < self.toler:
                break
            _iter += 1

    def get_item_error(self, out, y):
        minus = out - y
        return np.sum(np.multiply(minus, minus)) / 2.0

    def classifier(self, x):
        layer_size = len(self.layers)
        outs = []
        outs.append(np.mat(x).T)
        for i in range(1, layer_size):
            Oi = np.dot(self.weights[i - 1],
                        outs[i - 1]) + self.bias[i - 1]
            Oi = self.sigmoid(Oi)
            outs.append(Oi)
        return np.argmax(outs[-1])

    def sigmoid_prime(self, y):
        return np.multiply(y, (1 - y))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    data = pd.read_csv("../dataset/digit_recognizer/train.csv")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: x / 255.0)
    test_set = pd.read_csv("../dataset/digit_recognizer/test.csv")
    test_set = test_set.apply(lambda x: x / 255.0)
    test_set = test_set.values
    label_set = pd.get_dummies(data['label']).values
    train_set = data.iloc[:, 1:].values
    nn = neural_network(layers=[784, 64, 16, 10], alpha=0.1, toler=0.05,
                        max_iter=10000, active_func='sigmoid')
    nn.fit(train_set, label_set, method='MBGD', cost='cross_entropy')
    test_count = len(test_set)
    submission = []
    for i in range(test_count):
        label = nn.classifier(test_set[i])
        submission.append([i + 1, label])
    submission_df = pd.DataFrame(data=submission, columns=['ImageId', 'Label'])
    submission_df.to_csv(
        '../dataset/digit_recognizer/submission.csv', index=False)
