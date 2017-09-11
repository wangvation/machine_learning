#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class linear_classifier(object):
    """docstring for  linear_classifier"""

    def __init__(self, train_set, label_set):
        self.__train_set = train_set
        self.__target_mat = np.mat(label_set)
        self.__thet = None

    def fit(self, count=10000, step=0.001):
        m, n = shape(self.__train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self.train_set
        data = np.mat(data, dtype=np.float32)
        self.__thet = np.random.rand(n + 1, 1)
        for i in range(count):
            h = np.dot(data, self.__thet)
            err = h - self.__target_mat
            delta_thet = (1 / m) * step * np.dot(data.T, err)
            self.__thet -= delta_thet

    def classifier(self, item):
        item = [1] + item
        return np.dot(mat(item), self.__thet)


if __name__ == '__main__':
    main()
