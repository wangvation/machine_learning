#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from conv_layer import conv_layer
from fully_connect_layer import fully_connect_layer
from softmax_layer import softmax_layer
from pooling_layer import pooling_layer
import struct


class lenet_5(object):
    """docstring for lenet_5"""

    def __init__(self):
        self.layers = []
        self.layers.append(conv_layer(action=self.relu, zero_padding=2,
                                      action_derive=self.relu_prime,
                                      input_shape=(28, 28), kernel_stride=1,
                                      kernel_shape=(5, 5), kernel_num=6))
        self.layers.append(pooling_layer(input_shape=(6, 28, 28),
                                         kernel_shape=(6, 2, 2),
                                         pooling_type='max_pooling', stride=2))
        self.layers.append(conv_layer(action=self.relu, zero_padding=0,
                                      action_derive=self.relu_prime,
                                      input_shape=(6, 14, 14), kernel_stride=1,
                                      kernel_shape=(6, 5, 5), kernel_num=16))
        self.layers.append(pooling_layer(input_shape=(16, 10, 10),
                                         kernel_shape=(16, 2, 2),
                                         pooling_type='max_pooling', stride=2))
        self.layers.append(conv_layer(action=self.relu, zero_padding=0,
                                      action_derive=self.relu_prime,
                                      input_shape=(16, 5, 5), kernel_stride=1,
                                      kernel_shape=(16, 5, 5), kernel_num=120))
        self.layers.append(fully_connect_layer(action=self.relu,
                                               action_derive=self.relu_prime,
                                               layers=(120, 84)))
        self.layers.append(fully_connect_layer(action=self.relu,
                                               action_derive=self.relu_prime,
                                               layers=(84, 10)))
        self.layers.append(softmax_layer(10))

    def train(self, train_set, targets, alpha, method='SGD'):
        data_size, h, w = train_set.shape
        _iter = 0
        while _iter < 1000:
            batch_size, batch_index = self.get_batch(method, data_size)
            for i in batch_index:
                input_array = train_set[i, :, :]
                for layer in self.layers:
                    input_array = layer.forward(input_array)
                if _iter % 100 == 0:
                    print(self.cross_entropy(input_array, targets[i]))
                delta_map = self.layers[-1].backward(targets[i])
                for layer in self.layers[-2::-1]:
                    delta_map = layer.backward(delta_map)
            for layer in self.layers:
                layer.update(alpha, batch_size)
            _iter += 1
        pass

    def classfier(self, x):
        return np.argmax(x)

    def cross_entropy(self, y, targets):
        return -np.sum(np.multiply(targets, np.log(y)))

    def get_batch(self, method, data_size):
        if method == 'BGD':  # Batch gradient descent
            indexes = range(data_size)
            return data_size, indexes
        if method == 'SGD':  # Stochastic gradient descent
            indexes = [np.random.randint(low=0, high=data_size, size=1)]
            return 1, indexes
        if method == 'MBGD':  # Mini-batch gradient descent
            m = 30
            indexes = np.random.randint(low=0, high=data_size, size=m)
            return m, indexes

    def relu(self, array):
        negative_mask = array < 0
        array[negative_mask] = 0
        return array

    def relu_prime(self, y):
        return 1.0 if y > 0.0 else 0.0


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
    arr = np.zeros((len(data), 10), dtype=np.float32)
    arr[np.arange(len(data)), data] = 1
    # for i, one_hot in enumerate(data, 0):
    #     arr[i, one_hot] = 1.0
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
    lenet.train(train_set, train_labels, alpha=0.3, method='MBGD')
    test_count = len(test_set)
    right_count = 0
    for i in range(test_count):
        label = lenet.classifier(test_set[i])
        if 1 == test_labels[i, label]:
            right_count += 1
    print(right_count / test_count)


def do_kaggle():
    data = pd.read_csv("../dataset/digit_recognizer/train.csv")
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: x / 255.0)
    target_set = pd.get_dummies(data['label']).values
    data_size, columns_size = data.shape
    train_set = data.iloc[:, 1:].values
    train_set = train_set.reshape(42000, 28, 28)

    test_set = pd.read_csv("../dataset/digit_recognizer/test.csv")
    test_set = test_set.apply(lambda x: x / 255.0)
    test_set = test_set.values
    test_count = len(test_set)
    test_set = test_set.reshape(test_count, 28, 28)

    lenet = lenet_5()
    lenet.train(train_set, target_set, alpha=0.3, method='MSGD')
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
