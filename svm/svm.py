#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import random


class svm(object):
    """docstring for svm"""

    def __init__(self, C, max_iter, **kernel_opt):
        """y=wx+b"""
        self.kernel_opt = kernel_opt
        self.alphas = None
        self.omega = None
        self.kernel_mat = None
        self.errors = None
        self.support_vec = None
        self.max_iter = max_iter
        self.C = C
        self.bias = 0.0

    def fit(self, data_set, label_set):
        self.data_mat = np.mat(data_set)
        self.label_mat = np.mat(label_set).T
        self.data_size = len(label_set)
        self.kernel_mat = self.kernel_trans()
        self.smo()
        return self

    def get_bounds(self, alpha1, alpha2, label1, label2):
        if label1 == label2:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        return L, H

    def clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        if alpha > H:
            return H
        return alpha

    def get_new_bias(self, bias1, bias2, alpha1, alpha2):
        if 0 < alpha1 < self.C:
            return bias1
        if 0 < alpha2 < self.C:
            return bias2
        return (bias1 + bias2) / 2

    def clac_omega(self):
        self.omega = np.multiply(self.alphas.T, self.label_mat.T)

    def kernel(self, x, z):
        '''
        rbf kernel: exp(-||x-z||^2/(2*sigma^2))
        linear kernel:x.T*z
        '''
        func = self.kernel_opt['func']
        if func == 'rbf':
            sigma = self.kernel_opt['sigma']
            delta = x - z
            delta = delta * delta.T
            return math.exp(-delta / (2 * sigma**2))
        elif func == 'linear':
            return np.dot(x, z.T)

    def kernel_trans(self):
        m, n = np.shape(self.data_mat)
        kernel_mat = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                kernel_mat[i, j] = self.kernel(
                    self.data_mat[i, :], self.data_mat[j, :])
        return kernel_mat

    def update_alpha_bias(self, index1, index2):

        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]
        label1 = self.label_mat[index1]
        label2 = self.label_mat[index2]
        error1 = self.errors[index1]
        error2 = self.errors[index2]

        eta = self.kernel_mat[index1, index1] +\
            self.kernel_mat[index2, index2] -\
            2 * self.kernel_mat[index1, index2]

        alpha2_new = alpha2 + label2 * (error1 - error2) / eta
        L, H = self.get_bounds(alpha1, alpha2_new, label1, label2)
        if L == H:
            return 0
        self.alphas[index2] = self.clip_alpha(alpha2_new, L, H)
        if abs(alpha2_new - alpha2) < 0.00001:
            return 0
        alpha1_new = alpha1 + label1 * label2 * (alpha2_new - alpha2)
        self.alphas[index1] = alpha1_new

        b1_new = self.bias - error1 - label1 * \
            (alpha1_new - alpha1) * self.kernel_mat[index1, index1] - \
            label2 * (alpha2_new - alpha2) * self.kernel_mat[index1, index2]

        b2_new = self.bias - error2 - label1 * \
            (alpha1_new - alpha1) * self.kernel_mat[index1, index2] - \
            label2 * (alpha2_new - alpha2) * self.kernel_mat[index2, index2]

        self.bias = self.get_new_bias(b1_new, b2_new, alpha2, alpha2)
        return 1

    def is_meet_KKT(self, index):
        func_dist = float(self.label_mat[index, 0]) * \
            self.predict(index)
        if func_dist <= 1:
            return self.alphas[index] == self.C
        if func_dist == 1:
            return 0 < self.alphas[index] < self.C
        if func_dist >= 1:
            return self.alphas[index] == 0

    def select_second_alpha(self, first_index, valid_indexes):
        max_delta = 0
        second_index = -1
        error_1 = self.errors[first_index]
        for j in valid_indexes:
            if j is not first_index:
                error_j = self.errors[j]
                delta = abs(error_1 - error_j)
                if delta > max_delta:
                    max_delta = delta
                    second_index = j
        return second_index

    def update_errors(self):
        self.errors = (np.dot(self.omega, self.kernel_mat.T).T +
                       self.bias) - self.label_mat
        return self.errors

    def smo(self):
        self.bias = 0.0
        self.alphas = np.zeros((self.data_size, 1), dtype=np.float32)
        _iter = 0
        entier_flag = True
        update_num = 0
        self.clac_omega()
        self.update_errors()

        while (_iter < self.max_iter and update_num > 0) or entier_flag:
            update_num = 0
            if entier_flag:
                valid_indexes = range(self.data_size)
                for i in valid_indexes:
                    if self.is_meet_KKT(i):
                        continue
                    index_2 = self.select_second_alpha(i, valid_indexes)
                    if index_2 is -1:
                        continue
                    update_num += self.update_alpha_bias(i, index_2)
                entier_flag = False
            else:
                valid_indexes = np.nonzero(np.multiply(
                    self.alphas < self.C, self.alphas > 0))[1]
                for i in valid_indexes:
                    index_2 = self.select_second_alpha(
                        i, valid_indexes)
                    if index_2 is -1:
                        continue
                    update_num += self.update_alpha_bias(i, index_2)

            self.clac_omega()
            self.update_errors()
            _iter += 1
        self.get_support_vec()

    def get_support_vec(self):
        sv_index = np.nonzero(self.alphas > 0)[1]
        self.support_vec = self.data_mat[sv_index]
        return self.support_vec

    def predict(self, index):
        return float(np.dot(self.omega,
                            np.mat(self.kernel_mat[:, index]).T)) + self.bias

    def classfier(self, X):
        sv_index = np.nonzero(self.alphas > 0)[1]
        alphas = self.alphas[sv_index]
        labels = self.label_mat[sv_index]
        ret = 0
        for i in sv_index:
            ret += alphas[i] * labels[i] * self.kernel(self.support_vec[i], X)
        y = ret + self.bias
        return 1 if y >= 0 else -1


if __name__ == '__main__':
    data = pd.read_csv('../dataset/iris/bezdekIris.data')
    col_list = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'Class']

    setosa_versicolor = data[data.Class != 'Iris-virginica']
    setosa_virginica = data[data.Class != 'Iris-versicolor']
    versicolor_virginica = data[data.Class != 'Iris-setosa']

    setosa_versicolor[col_list[-1]] = \
        setosa_versicolor[col_list[-1]].apply(
        lambda x: 1 if x == 'Iris-versicolor' else -1)
    setosa_virginica[col_list[-1]] = \
        setosa_virginica[col_list[-1]].apply(
        lambda x: 1 if x == 'Iris-virginica' else -1)
    versicolor_virginica[col_list[-1]] = \
        versicolor_virginica[col_list[-1]].apply(
        lambda x: 1 if x == 'Iris-versicolor' else -1)

    result_dict = {"setosa_versicolor": {-1: "Iris-setosa",
                                         1: "Iris-versicolor"},
                   "setosa_virginica": {-1: "Iris-setosa",
                                        1: "Iris-virginica"},
                   "versicolor_virginica": {-1: "Iris-virginica",
                                            1: "Iris-versicolor"}
                   }

    svm_setosa_versicolor = svm(C=100, max_iter=10000, func='linear')
    svm_setosa_versicolor.fit(setosa_versicolor[col_list[:-1]].values,
                              setosa_versicolor[col_list[-1]].values,)
    svm_setosa_virginica = svm(C=100, max_iter=10000, func='linear')\
        .fit(setosa_virginica[col_list[:-1]].values,
             setosa_virginica[col_list[-1]].values)
    svm_versicolor_virginica = svm(C=100, max_iter=10000, func='linear')\
        .fit(versicolor_virginica[col_list[:-1]].values,
             versicolor_virginica[col_list[-1]].values)

    svms = {"setosa_versicolor": svm_setosa_versicolor,
            "setosa_virginica": svm_setosa_virginica,
            "versicolor_virginica": svm_versicolor_virginica}
    test_index = [random.randint(0, 149) for _ in range(20)]
    right_num = 0
    for index in test_index:
        class_dict = {
            'Iris-setosa': 0,
            'Iris-virginica': 0,
            'Iris-versicolor': 0
        }
        test_data = data.iloc[index]
        for name, _svm in svms.items():
            result_label = _svm.classfier(test_data[col_list[:-1]].values)
            result_class = result_dict[name][result_label]
            class_dict[result_class] += 1
            label = None
            max_vote = 0
            for _class, value in class_dict.items():
                if value > max_vote:
                    max_vote = value
                    label = _class
            if label == test_data[col_list[-1]]:
                right_num += 1
    print(right_num)