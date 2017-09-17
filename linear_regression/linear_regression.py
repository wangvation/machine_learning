#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
sys.path.append("../common")
from csv_utils import *


class linear_classifier(object):
    """docstring for  linear_classifier"""

    def __init__(self, train_set, label_set, algorithm='gradient_descent'):
        self.__train_set = train_set
        self.__target_mat = np.mat(label_set).T
        self.__algorithm = algorithm
        self.__thet = None

    def fit(self):
        if self.__algorithm == 'least_squares':
            self.least_squares()
        elif self.__algorithm == 'gradient_descent':
            self.gradient_descent(iterations=10000, alpha=0.0003)

    def gradient_descent(self, iterations=10000, alpha=0.001):
        # gradient_descent
        m, n = np.shape(self.__train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self.__train_set
        data = np.mat(data, dtype=np.float32)
        self.__thet = np.random.rand(n + 1, 1) - 0.5
        for i in range(iterations):
            h = np.dot(data, self.__thet)
            err = h - self.__target_mat
            delta_thet = (1 / m) * alpha * np.dot(data.T, err)
            self.__thet = self.__thet - delta_thet
        pass

    def least_squares(self):
        '''
        w=(X.T*X).I*X.T*y
        '''
        m, n = np.shape(self.__train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self.__train_set
        data = np.mat(data, dtype=np.float32)
        self.__thet = np.dot(
            np.dot(np.dot(data.T, data).I, data.T), self.__target_mat)
        pass

    def classifier(self, item):
        item = [1] + item
        return np.dot(np.mat(item), self.__thet)

    def get_thet(self):
        return self.__thet


if __name__ == '__main__':
    data = load_csv('../dataset/machine/machine.data')
    tartget = [x[8] for x in data]
    data = [x[2:] for x in data]
    train_set = []
    for row in data:
        del row[-2]
    for row in data:
        train_set.append([float(item) for item in row])
    tartget = [float(x) for x in tartget]
    # 最小二乘法
    classifier1 = linear_classifier(
        train_set, tartget, algorithm='least_squares')
    classifier1.fit()
    print(classifier1.classifier(train_set[191]))
    # 梯度下降
    classifier2 = linear_classifier(
        train_set, tartget, algorithm='gradient_descent')
    classifier2.fit()
    print(classifier2.classifier(train_set[191]))
