#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math


class svm(object):
    """docstring for svm"""

    def __init__(self, data_mat, label_set):
        """y=wx+b"""
        self.__data_mat = data_mat
        self.__label_mat = np.mat(label_set)
        self.__data_size = len(label_set)
        self.__alphas = None
        self.__omega = None
        self.__kernel_mat = None
        self.__errors = None
        self.__C = 0
        self.__bias = 0.0

    def fit(self, **kernel_opt):
        self.__kernel_mat = self.kernel_trans(kernel_opt)

    def get_bounds(self, alpha1, alpha2, label1, label2):
        if label1 == label2:
            L = max(0, alpha2 + alpha1 - self.__C)
            H = min(self.__C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.__C, self.__C + alpha2 - alpha1)
        return L, H

    def clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        if alpha > H:
            return H
        return alpha

    def get_new_bias(self, bias1, bias2, alpha1, alpha2):
        if 0 < alpha1 < self.__C:
            return bias1
        if 0 < alpha2 < self.__C:
            return bias2
        return (bias1 + bias2) / 2

    def clac_omega(self):
        self.__omega = np.multiply(self.__alphas, self.__label_mat)

    def kernel(self, x, z, **kernel_opt):
        '''
        rbf kernel: exp(-||x-z||^2/(2*sigma^2))
        linear kernel:x.T*z
        '''
        func = kernel_opt['func']
        if func == 'rbf':
            sigma = kernel_opt['sigma']
            delta = x - z
            delta = delta * delta.T
            return math.exp(-delta / (2 * sigma**2))
        elif func == 'linear':
            return np.dot(x, z.T)

    def kernel_trans(self, **kernel_opt):
        m, n = np.shape(self.data_mat)
        kernel_mat = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                kernel_mat[i, j] = self.kernel(
                    self.data_mat[i, :], self.data_mat[j, :], kernel_opt)
        return kernel_mat

    def update_alpha_bias(self, index1, index2):

        alpha1 = self.__alphas[index1]
        alpha2 = self.__alphas[index2]
        label1 = self.__label_mat[index1]
        label2 = self.__label_mat[index2]
        error1 = self.__errors[index1]
        error2 = self.__errors[index2]

        eta = self.kernel_mat[index1, index1] +\
            self.kernel_mat[index2, index2] -\
            2 * self.kernel_mat[index1, index2]

        alpha2_new = alpha2 + label2 * (error1 - error2) / eta
        L, H = self.get_bounds(alpha1, alpha2_new, label1, label2)
        self.__alphas[index2] = self.clip_alpha(alpha2_new, L, H)
        alpha1_new = alpha1 + label1 * label2 * (alpha2_new - alpha2)
        self.__alphas[index1] = alpha1_new

        b1_new = self.__bias - error1 - label1 * \
            (alpha1_new - alpha1) * self.kernel_mat[index1, index1] - \
            label2 * (alpha2_new - alpha2) * self.kernel_mat[index1, index2]

        b2_new = self.__bias - error2 - label1 * \
            (alpha1_new - alpha1) * self.kernel_mat[index1, index2] - \
            label2 * (alpha2_new - alpha2) * self.kernel_mat[index2, index2]

        self.__bias = self.get_new_bias(b1_new, b2_new, alpha2, alpha2)

    def is_meet_KKT(self, index):
        func_dist = self.__label_mat[index] * \
            self.predict(self.data_mat[index])
        if func_dist <= 1:
            return self.__alphas[index] == self.C
        if func_dist == 1:
            return 0 < self.__alphas[index] < self.C
        if func_dist >= 1:
            return self.__alphas[index] == 0

    def select_pair(self, first_index, valid_indexes):
        max_delta = 0
        second_index = -1
        if not self.is_meet_KKT(self.__C, first_index):
            error_1 = self.__errors[first_index]
            for j in valid_indexes:
                if j is not first_index and self.is_meet_KKT(self.__C, j):
                    error_j = self.__errors[j]
                    delta = abs(error_1 - error_j)
                    if delta > max_delta:
                        max_delta = delta
                        second_index = j
        return first_index, second_index

    def update_errors(self):
        self.__errors = (np.dot(self.__omega, self.__kernel_mat.T) +
                         self.__bias) - self.__label_mat
        return self.__errors

    def smo(self, toler, maxIter):
        self.__bias = 0.0
        self.__alphas = np.zeros((self.__data_size, 1), dtype=np.float32)
        _iter = 0
        entier_flag = True
        alpha_update_num = 0
        self.calc_omega(self.__alphas)
        self.update_errors()

        while entier_flag or (_iter < maxIter and alpha_update_num > 0):
            alpha_update_num = 0
            if entier_flag:
                valid_indexes = range(self.__data_size)
                entier_flag = False
            else:
                valid_indexes = np.nonzero(np.multiply(
                    self.__alphas < self.C, self.__alphas > 0))[1]

            for i in valid_indexes:
                index_1, index_2 = self.select_pair(i, valid_indexes)
                if index_2 is -1:
                    continue
                self.update_alpha_bias(index_1, index_2)
                alpha_update_num += 1

            self.clac_omega()
            self.update_errors()
            _iter += 1

    def predict(self, X):
        kernels = np.mat([self.kernel(item, X)] for item in self.data_mat)
        return np.dot(self.__omega, kernels.T) + self.__bias

    def classfier(self, X):
        return 1 if self.predict(X) >= 0 else -1


if __name__ == '__main__':
    data = pd.read_csv('ddd')
    pass
