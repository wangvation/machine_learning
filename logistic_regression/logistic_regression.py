#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class logistic_classifier(object):
    """docstring for  logistic_classifier"""

    def __init__(self, train_set, label_set):
        self.__train_set = train_set
        self.__target_mat = np.mat(label_set).T
        self.__thet = None
        self.__errores = []

    def fit(self, iterations=100000, alpha=0.03):
        self.__gradient_descent(iterations, alpha)

    def __gradient_descent(self, iterations, alpha):
        # gradient_descent
        m, n = np.shape(self.__train_set)
        data = np.ones((m, n + 1))
        data[:, 1:] = self.__train_set
        data = np.mat(data, dtype=np.float32)
        self.__thet = np.random.rand(n + 1, 1)
        for i in range(iterations):
            h = self.__logsitic(np.dot(data, self.__thet))
            err = h - self.__target_mat
            if i % (iterations / 100) == 0:
                self.__errores.append(np.dot(err.T, err))
            delta_thet = (1.0 / m) * alpha * np.dot(data.T, err)
            self.__thet = self.__thet - delta_thet

    def __logsitic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def classifier(self, input_data):
        m, n = np.shape(input_data)
        tmp = np.ones((m, n + 1))
        tmp[:, 1:] = input_data
        result = self.__logsitic(np.dot(tmp, self.__thet))

        def func(x):
            return 1 if x >= 0.5 else 0
        return [func(x) for x in result]

    def get_thet(self):
        return self.__thet

    def get_errores(self):
        return self.__errores


if __name__ == '__main__':
    data = pd.read_csv(
        '../dataset/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    col_list = ['id', 'Clump_Thickness', 'Uniformity_of_Cell_Size',
                'Uniformityof_Cell_Shape', 'Marginal_Adhesion',
                'Single_Epithelial', 'Bare_Nuclei', 'Bland_Chromatin',
                'Normal_Nucleoli', 'Mitoses', 'Class']
    # remove the abnormal data
    data = data[data.Bare_Nuclei != '?']
    # 0 for benign, 1 for malignant
    data[col_list[-1]] = data[col_list[-1]].apply(lambda x: 1 if x == 4 else 0)
    data_size = len(data.values)
    x = data_size / 10
    data_index = range(data_size)
    # k - fold CrossValidation
    K = 10
    errors = []
    rights = []
    for i in range(K):
        test_index = range(i * x, (i + 1) * x)
        train_index = list(set(data_index) ^ set(test_index))
        test_set = data[col_list[1:-1]].iloc[test_index].values
        test_label = data[col_list[-1]].iloc[test_index].values
        train_set = data[col_list[1:-1]].iloc[train_index].values
        label_set = data[col_list[-1]].iloc[train_index].values
        classifier = logistic_classifier(train_set, label_set)
        classifier.fit(iterations=100000, alpha=0.03)
        result_set = classifier.classifier(test_set)
        err = 0
        right = 0
        for predict, label in zip(result_set, test_label):
            if predict != label:
                err += 1
            else:
                right += 1

        errors.append(err * 1.0 / len(test_set))
        rights.append([right, len(test_set)])
    print(np.mean(errors))
    print(rights)
