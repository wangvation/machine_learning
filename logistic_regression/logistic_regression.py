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

    def __init__(self, train_set, label_set):

        self.__train_set = train_set
        self.__target_mat = np.mat(label_set).T
        self.__thet = None

    def fit(self):
       # gradient_descent
        m, n = np.shape(self.__train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self.__train_set
        data = np.mat(data, dtype=np.float32)
        self.__thet = np.random.rand(n + 1, 1)
        for i in range(iterations):
            h = np.dot(data, self.__thet)
            err = h - self.__target_mat
            delta_thet = (1 / m) * alpha * np.dot(data.T, err)
            self.__thet = self.__thet - delta_thet
        pass

    def classifier(self, item):
        item = [1] + item
        return np.dot(np.mat(item), self.__thet)

    def get_thet(self):
        return self.__thet


if __name__ == '__main__':
