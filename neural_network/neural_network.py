#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import copy


class neural_network(object):
    """docstring for neural_network"""

    def __init__(self, layers=[], alpha=0.3, toler=0.1,
                 max_iter=10000, lamda=0.0, active_func='sigmoid',
                 method='BGD', cost='MSE'):
        self.layers = layers
        self.alpha = alpha
        self.toler = toler
        self.max_iter = max_iter
        self.lamda = lamda
        if active_func == 'sigmoid':
            self.active = self.sigmoid
            self.active_derive = self.sigmoid_derive
        else:
            self.active = self.tanh
            self.active_derive = self.tanh_derive
        self.method = method
        self.item_cost = (self.mean_square_error if cost ==
                          'MSE' else self.cross_entropy)
        self.num_layers = len(layers)
        self.weights = []
        self.bias = []
        for i in range(1, self.num_layers):
            n = self.layers[i - 1]
            m = self.layers[i]
            self.weights.append(np.random.normal(loc=0.0,
                                                 scale=1.0 / np.sqrt(m * n),
                                                 size=(m, n)))
            self.bias.append(np.zeros((m, 1)))

    def fit(self, data_set, target_set, debug=False):
        self.data_mat = np.mat(data_set)
        self.target_mat = np.mat(target_set)
        self.train(debug)

    def train(self, debug):
        data_size, _ = np.shape(self.data_mat)
        _iter = 0
        while _iter < self.max_iter:
            indexes = self.get_batch()
            outs = self.forward(indexes, self.weights, self.bias)
            targets = [self.target_mat[i].T for i in indexes]
            predict = [out[-1] for out in outs]
            batch_size = len(indexes)
            if debug:
                self.gradient_check(outs, targets, indexes)
                debug = False
            errors = self.cost(batch_size, predict, targets)
            if _iter % 100 == 0:
                self.alpha *= 0.98
                print(errors, self.alpha)
            if errors < self.toler:
                break
            self.gradient_descent(outs, targets, batch_size)
            _iter += 1

    def get_batch(self):
        data_size, _ = np.shape(self.data_mat)
        if self.method == 'BGD':  # Batch gradient descent
            indexes = range(data_size)
        elif self.method == 'SGD':  # Stochastic gradient descent
            indexes = [random.randint(0, data_size - 1)]
        elif self.method == 'MBGD':  # Mini-batch gradient descent
            m = 20
            indexes = [random.randint(0, data_size - 1) for x in range(m)]
        return indexes

    def forward(self, indexes, weights, bias):
        outs = []
        for i in indexes:
            actives = []
            actives.append(self.data_mat[i].T)
            for j in range(1, self.num_layers):
                Oj = np.dot(weights[j - 1], actives[j - 1]) + bias[j - 1]
                actives.append(self.active(Oj))
            outs.append(actives)
        return outs

    def gradient_descent(self, outs, targets, batch_size):
        weights_delta = [np.zeros(w.shape) for w in self.weights]
        bias_delta = [np.zeros(b.shape) for b in self.bias]
        for target, out in zip(targets, outs):
            error = self.item_cost(out[-1], target)
            if error <= 0.01:
                continue
            weights_grad, bias_grad = self.back_propagation(out, target)
            for k in range(self.num_layers - 1):
                weights_delta[k] += weights_grad[k]
                bias_delta[k] += bias_delta[k]
        for k in range(self.num_layers - 1):
            self.weights[k] -= self.alpha * \
                (weights_delta[k] + self.lamda * self.weights[k]) / batch_size
            self.bias[k] -= self.alpha * bias_delta[k] / batch_size

    def back_propagation(self, out, target):
        output_layer = self.num_layers - 1
        if self.item_cost == self.cross_entropy:
            delta = out[output_layer] - target
        else:
            delta = -np.multiply(target - out[output_layer],
                                 self.active_derive(out[output_layer]))
        layer = output_layer - 1
        weights_grad = [None for _ in range(self.num_layers - 1)]
        bias_grad = [None for _ in range(self.num_layers - 1)]
        while layer >= 0:
            weights_grad[layer] = np.dot(delta, out[layer].T)
            bias_grad[layer] = delta
            delta = np.multiply(np.dot(self.weights[layer].T, delta),
                                self.active_derive(out[layer]))
            layer -= 1
        return weights_grad, bias_grad

    def gradient_check(self, outs, targets, indexes):
        weights = copy.deepcopy(self.weights)
        bias = copy.deepcopy(self.bias)
        target, out, k = targets[0], outs[0], indexes[0]
        weights_grad, bias_grad = self.back_propagation(out, target)
        epsilon = 10e-4
        for layer_index in range(1, self.num_layers):
            weight = weights[layer_index - 1]
            m, n = np.shape(weight)
            for i in range(m):
                for j in range(n):
                    weight[i, j] += epsilon
                    error1 = self.item_cost(
                        self.predict(self.data_mat[k],
                                     weights, bias), target)
                    weight[i, j] -= 2 * epsilon
                    error2 = self.item_cost(
                        self.predict(self.data_mat[k],
                                     weights, bias), target)
                    weight[i, j] += epsilon
                    print(weights_grad[layer_index - 1][i, j],
                          (error1 - error2) / (2 * epsilon))

    def cost(self, data_size, outs, targets):
        errors = 0
        for out, y in zip(outs, targets):
            errors += self.item_cost(out, y)
        errors = errors / data_size
        weight_decay = 0
        for weight in self.weights:
            weight_decay += np.sum(np.multiply(weight, weight))
        weight_decay = weight_decay * self.lamda / 2
        errors = errors + weight_decay
        return errors

    def predict(self, x, weights, bias):
        out = np.mat(x).T
        for i in range(1, self.num_layers):
            out = np.dot(weights[i - 1], out) + bias[i - 1]
            out = self.sigmoid(out)
        return out

    def classifier(self, x):
        out = self.predict(x, self.weights, self.bias)
        return np.argmax(out)

    def cross_entropy(self, out, y):
        return np.sum(-np.multiply(y, np.log(out)) -
                      np.multiply(1 - y, np.log(1 - out)))

    def mean_square_error(self, out, y):
        minus = out - y
        return np.sum(np.multiply(minus, minus)) / 2.0

    def softmax(self, x):
        _sum = np.sum(x)
        return x / _sum

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derive(self, y):
        return np.multiply(y, (1 - y))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derive(self, y):
        return 1 - np.multiply(y, y)


if __name__ == '__main__':
    data = pd.read_csv("../dataset/digit_recognizer/train.csv")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: x / 255.0)
    debug = True
    if debug:
        data_size, columns_size = data.shape
        train_set = data.iloc[:, 1:].values
        label_set = pd.get_dummies(data['label']).values
        all_mask = np.repeat(True, data_size)
        test_index = np.random.randint(0, data_size, size=data_size // 10,
                                       dtype=np.int32)

        test_label = label_set[test_index, :]
        test_set = train_set[test_index, :]

        all_mask[test_index] = False
        train_set = train_set[all_mask, :]
        target_set = label_set[all_mask, :]

    else:
        target_set = pd.get_dummies(data['label']).values
        train_set = data.iloc[:, 1:].values

        test_set = pd.read_csv("../dataset/digit_recognizer/test.csv")
        test_set = test_set.apply(lambda x: x / 255.0)
        test_set = test_set.values

    nn = neural_network(layers=[784, 100, 10], alpha=1.0, toler=0.01,
                        max_iter=10000, lamda=0.001, active_func='sigmoid',
                        method='MBGD', cost='cross_entropy')

    if debug:
        nn.fit(train_set, target_set, False)
        test_count = len(test_set)
        right_count = 0
        for i in range(test_count):
            label = nn.classifier(test_set[i])
            if 1 == test_label[i, label]:
                right_count += 1
        print(right_count / test_count)
    else:
        nn.fit(train_set, target_set)
        test_count = len(test_set)
        submission = []
        for i in range(test_count):
            label = nn.classifier(test_set[i])
            submission.append([i + 1, label])
        submission_df = pd.DataFrame(data=submission,
                                     columns=['ImageId', 'Label'])
        submission_df.to_csv(
            '../dataset/digit_recognizer/submission.csv', index=False)
