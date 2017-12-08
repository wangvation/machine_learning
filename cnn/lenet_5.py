#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from conv_layer import conv_layer
from fully_connect_layer import fully_connect_layer
from softmax_layer import softmax_layer
from pooling_layer import pooling_layer


class lenet_5(object):
    """docstring for lenet_5"""

    def __init__(self, alpha):
        self.layers = []
        self.layers.append(conv_layer(active=self.relu, zero_padding=2,
                                      active_derive=self.relu_prime,
                                      input_shape=(28, 28), kernel_stride=1,
                                      kernel_shape=(5, 5), kernel_num=6))
        self.layers.append(pooling_layer(input_shape=(6, 28, 28),
                                         kernel_shape=(6, 2, 2),
                                         pooling_type='max_pooling', stride=2))
        self.layers.append(conv_layer(active=self.relu, zero_padding=0,
                                      active_derive=self.relu_prime,
                                      input_shape=(6, 14, 14), kernel_stride=1,
                                      kernel_shape=(6, 5, 5), kernel_num=16))
        self.layers.append(pooling_layer(input_shape=(16, 10, 10),
                                         kernel_shape=(16, 2, 2),
                                         pooling_type='max_pooling', stride=2))
        self.layers.append(conv_layer(active=self.relu, zero_padding=0,
                                      active_derive=self.relu_prime,
                                      input_shape=(16, 5, 5), kernel_stride=1,
                                      kernel_shape=(16, 5, 5), kernel_num=120))
        self.layers.append(fully_connect_layer(active=self.relu,
                                               active_derive=self.relu_prime,
                                               layers=(120, 84)))
        self.layers.append(fully_connect_layer(active=self.relu,
                                               active_derive=self.relu_prime,
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
                delta_map = self.layers[-1].backward(targets[i])
                for layer in self.layers[-2::-1]:
                    delta_map = layer.backward(delta_map)
            for layer in self.layers:
                layer.update(alpha, batch_size)
            _iter += 1
        pass

    def get_batch(self, method, data_size):
        data_size, _ = np.shape(self.data_mat)
        if method == 'BGD':  # Batch gradient descent
            indexes = range(data_size)
            return data_size, indexes
        if method == 'SGD':  # Stochastic gradient descent
            indexes = [np.random.randint(low=0, high=data_size, size=1)]
            return 1, indexes
        if method == 'MBGD':  # Mini-batch gradient descent
            m = 10
            indexes = np.random.randint(low=0, high=data_size, size=m)
            return m, indexes

    def relu(self, x):
        return np.max(0, x)

    def relu_prime(self, y):
        return 1.0 if y > 0.0 else 0.0


def main():
    pass


if __name__ == '__main__':
    main()
