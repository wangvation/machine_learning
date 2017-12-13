#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from conv_layer import conv_layer
from fully_connect_layer import fc_layer
from softmax_layer import softmax_layer
from pooling_layer import pooling_layer
import struct


class lenet_5(object):
    """docstring for lenet_5"""

    def __init__(self):
        self.layers = []
        self.layers.append(conv_layer(action=self.relu, zero_padding=2,
                                      action_derive=self.relu_derive,
                                      input_shape=(28, 28), kernel_stride=1,
                                      kernel_shape=(5, 5), kernel_num=6))
        self.layers.append(pooling_layer(input_shape=(6, 28, 28),
                                         kernel_shape=(6, 2, 2),
                                         pooling_type='max_pooling', stride=2))
        self.layers.append(conv_layer(action=self.relu, zero_padding=0,
                                      action_derive=self.relu_derive,
                                      input_shape=(6, 14, 14), kernel_stride=1,
                                      kernel_shape=(6, 5, 5), kernel_num=16))
        self.layers.append(pooling_layer(input_shape=(16, 10, 10),
                                         kernel_shape=(16, 2, 2),
                                         pooling_type='max_pooling', stride=2))
        self.layers.append(conv_layer(action=self.relu, zero_padding=0,
                                      action_derive=self.relu_derive,
                                      input_shape=(16, 5, 5), kernel_stride=1,
                                      kernel_shape=(16, 5, 5), kernel_num=120))
        self.layers.append(fc_layer(action=self.relu,
                                    action_derive=self.relu_derive,
                                    layers=(120, 84)))
        self.layers.append(softmax_layer(action=self.relu,
                                         action_derive=self.relu_derive,
                                         layers=(84, 10)))

    def train(self, train_set, targets, alpha, method='SGD'):
        data_size, h, w = train_set.shape
        _iter = 0
        while _iter < 1000:
            batch_size, batch_index = self.get_batch(method, data_size)
            errors = 0
            for i in batch_index:
                output = train_set[i, ...]
                target = targets[i]
                for layer in self.layers:
                    output = layer.forward(output)
                error = self.layers[-1].get_error(target)
                if error < 0.01:
                    print('error', error)
                    continue
                errors += error
                delta_map = self.layers[-1].backward(target)
                for layer in self.layers[-2::-1]:
                    delta_map = layer.backward(delta_map)
            print('errors', errors / batch_size)
            for layer in self.layers:
                layer.update(alpha, batch_size)
            _iter += 1
            # break
        pass

    def classifier(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return np.argmax(out)

    def mean_square_error(self, y, target):
        minus = y - target
        return np.sum(np.multiply(minus, minus)) / 2.0

    def get_batch(self, method, data_size):
        if method == 'BGD':  # Batch gradient descent
            indexes = range(data_size)
            return data_size, indexes
        if method == 'SGD':  # Stochastic gradient descent
            indexes = [random.randint(0, data_size)]
            return 1, indexes
        if method == 'MBGD':  # Mini-batch gradient descent
            m = 10
            indexes = [random.randint(0, data_size) for _ in range(m)]
            return m, indexes

    def relu(self, array):
        array[array < 0] = 0.0
        return array

    def relu_derive(self, y):
        derive = np.zeros(y.shape)
        derive[y > 0] = 1.0
        return derive

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def softplus_derive(self, y):
        exp_y = np.exp(y)
        return exp_y / (exp_y + 1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derive(self, y):
        return np.multiply(y, (1 - y))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derive(self, y):
        return 1.0 - np.power(y, 2)


def load_mnist(images_path, labels_path):
    """Load MNIST data from `path`"""
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 28, 28)

    return images, labels


def normalize(array):
    return array.astype(np.float32) / 255.0


def one_hot(data):
    arr = np.zeros((len(data), 10, 1))
    arr[np.arange(len(data)), data] = 1.0
    return arr


def do_mnist():
    train_set, train_labels = load_mnist(
        '../dataset/mnist/train-images-idx3-ubyte',
        '../dataset/mnist/train-labels-idx1-ubyte')
    test_set, test_labels = load_mnist(
        '../dataset/mnist/t10k-images-idx3-ubyte',
        '../dataset/mnist/t10k-labels-idx1-ubyte')
    train_set = normalize(train_set)
    train_labels = one_hot(train_labels)
    test_set = normalize(test_set)
    test_labels = one_hot(test_labels)
    lenet = lenet_5()
    lenet.train(train_set, train_labels, alpha=0.01, method='SGD')
    test_count = len(test_set)
    right_count = 0
    for i in range(100):
        label = lenet.classifier(test_set[i, ...])
        if 1 == test_labels[i, label]:
            right_count += 1
    print(right_count / test_count)


def do_kaggle():
    data = pd.read_csv("../dataset/digit_recognizer/train.csv")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: x / 255.0)
    target_set = pd.get_dummies(data['label']).values
    data_size, columns_size = data.shape
    train_set = data.iloc[:, 1:].values
    train_set = train_set.reshape(data_size, 28, 28)

    test_set = pd.read_csv("../dataset/digit_recognizer/test.csv")
    test_set = test_set.apply(lambda x: x / 255.0)
    test_set = test_set.values
    test_count = len(test_set)
    test_set = test_set.reshape(test_count, 28, 28)

    lenet = lenet_5()
    lenet.train(train_set, target_set, alpha=0.3, method='MBGD')
    submission = []
    for i in range(test_count):
        label = lenet.classifier(test_set[i])
        submission.append([i + 1, label])
    submission_df = pd.DataFrame(data=submission,
                                 columns=['ImageId', 'Label'])
    submission_df.to_csv(
        '../dataset/digit_recognizer/submission.csv', index=False)


if __name__ == '__main__':
    debug = True
    # debug = False
    if debug:
        do_mnist()
    else:
        do_kaggle()
