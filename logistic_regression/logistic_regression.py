#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
sys.path.append("../common")
from csv_utils import *


class logistic_classifier(object):
    """docstring for  logistic_classifier"""

    def __init__(self, train_set, label_set, algorithm='gradient_descent'):
        self.__train_set = train_set
        self.__target_mat = np.mat(label_set).T
        self.__algorithm = algorithm
        self.__thet = None
        self.__errores = []

    def fit(self):
        self.gradient_descent(iterations=100000, alpha=0.03)

    def gradient_descent(self, iterations=100000, alpha=0.03):
        # gradient_descent
        m, n = np.shape(self.__train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self.__train_set
        data = np.mat(data, dtype=np.float32)
        self.__thet = np.random.rand(n + 1, 1)
        for i in range(iterations):
            h = np.dot(data, self.__thet)
            err = h - self.__target_mat
            if i % (iterations / 100) == 0:
                self.__errores.append(np.dot(err.T, err))
            delta_thet = (1.0 / m) * alpha * np.dot(data.T, err)
            self.__thet = self.__thet - delta_thet
        pass

    def logsitic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def classifier(self, item):
        n = len(item)
        tmp = np.ones((1, n + 1))
        tmp[0, 1:] = item
        return np.dot(tmp, self.__thet)

    def get_thet(self):
        return self.__thet

    def get_errores(self):
        return self.__errores


if __name__ == '__main__':
