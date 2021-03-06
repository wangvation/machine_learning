#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearClassifier(object):
    """docstring for  LinearClassifier"""

    def __init__(self, train_set, label_set):
        self._train_set = train_set
        self._target_mat = np.mat(label_set).T
        self._thet = None
        self._errores = []

    def fit(self, algorithm='gradient_descent', iterations=100000, alpha=0.03):
        if algorithm == 'least_squares':
            self._least_squares()
        elif algorithm == 'gradient_descent':
            self._gradient_descent(iterations, alpha)

    def __gradient_descent(self, iterations=100000, alpha=0.03):
        # gradient_descent
        m, n = np.shape(self._train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self._train_set
        data = np.mat(data, dtype=np.float32)
        self._thet = np.random.rand(n + 1, 1)
        for i in range(iterations):
            h = np.dot(data, self._thet)
            err = h - self._target_mat
            if i % (iterations / 100) == 0:
                self._errores.append(np.dot(err.T, err))
            delta_thet = (1.0 / m) * alpha * np.dot(data.T, err)
            self._thet = self._thet - delta_thet
        pass

    def __least_squares(self):
        '''
        w=(X.T*X).I*X.T*y
        '''
        m, n = np.shape(self._train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self._train_set
        data = np.mat(data, dtype=np.float32)
        self._thet = np.dot(
            np.dot(np.dot(data.T, data).I, data.T), self._target_mat)
        pass

    def classifier(self, item):
        n = len(item)
        tmp = np.ones((1, n + 1))
        tmp[0, 1:] = item
        return np.dot(tmp, self._thet)

    def get_thet(self):
        return self._thet

    def get_errores(self):
        return self._errores


if __name__ == '__main__':
    data = pd.read_csv('../dataset/machine/machine.data')
    _xdata = data['PRP'].copy()
    collist = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP']
    for col in collist:
        data[col] = data[col].apply(lambda x: float(x))
        min = data[col].min()
        max = data[col].max()
        # min_max_normalize
        data[col] = data[col].apply(lambda x: (
            x - min) / (max - min))
    train_set = data[collist[:-1]].values
    tartget = data[collist[-1]].values
    classifier1 = LinearClassifier(
        train_set, tartget)
    classifier1.fit(algorithm='least_squares')
    print('--------least_squares--------')
    print(classifier1.get_thet())
    classifier2 = LinearClassifier(
        train_set, tartget)
    classifier2.fit(algorithm='gradient_descent',
                    iterations=100000, alpha=0.03)
    print('-------gradient_descent------')
    print(classifier2.get_thet())
    y = np.array(classifier2.get_errores()).reshape(1, 100)[0]
    x = np.arange(100)
    plt.plot(x, y)
    plt.show()
