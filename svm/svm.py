#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math


class svm(object):
    """docstring for svm"""

    def __init__(self, C, **kernel_opt):
        """y=wx+b"""
        self.kernel_opt = kernel_opt
        self.alphas = None
        self.omega = None
        self.kernel_mat = None
        self.errors = None
        self.support_vec = None
        self.C = C
        self.bias = 0.0

    def fit(self, data_mat, label_set):
        self.data_mat = data_mat
        self.label_mat = np.mat(label_set)
        self.data_size = len(label_set)
        self.kernel_mat = self.kernel_trans(self.kernel_opt)

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
        sv_index = np.nonzero(self.alphas > 0)[1]
        self.omega = np.multiply(
            self.alphas[sv_index], self.label_mat[sv_index])

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

    def update_alpha_bias(self, toler, index1, index2):

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
        func_dist = self.label_mat[index] * \
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
        self.errors = (np.dot(self.omega, self.kernel_mat.T) +
                       self.bias) - self.label_mat
        return self.errors

    def smo(self, toler, maxIter):
        self.bias = 0.0
        self.alphas = np.zeros((self.data_size, 1), dtype=np.float32)
        _iter = 0
        entier_flag = True
        update_num = 0
        self.calc_omega(self.alphas)
        self.update_errors()

        while (_iter < maxIter and update_num > 0) or entier_flag:
            update_num = 0
            if entier_flag:
                valid_indexes = range(self.data_size)
                for i in valid_indexes:
                    if self.is_meet_KKT(i):
                        continue
                    index_2 = self.select_second_alpha(i, valid_indexes)
                    if index_2 is -1:
                        continue
                    update_num += self.update_alpha_bias(toler, i, index_2)
                entier_flag = False
            else:
                valid_indexes = np.nonzero(np.multiply(
                    self.alphas < self.C, self.alphas > 0))[1]
                for i in valid_indexes:
                    index_2 = self.select_second_alpha(
                        i, valid_indexes)
                    if index_2 is -1:
                        continue
                    update_num += self.update_alpha_bias(toler, i, index_2)

            self.clac_omega()
            self.update_errors()
            _iter += 1
        self.get_support_vec()

    def get_support_vec(self):
        sv_index = np.nonzero(self.alphas > 0)[1]
        self.support_vec = self.data_mat[sv_index]
        return self.support_vec

    def predict(self, index):
        return np.dot(self.omega, self.kernel_mat[:, index]) + self.bias

    def classfier(self, X):
        kernels = np.mat([self.kernel(sv, X) for sv in self.support_vec])
        y = np.dot(self.omega, kernels.T) + self.bias
        return 1 if y >= 0 else -1


if __name__ == '__main__':
    data = pd.read_csv('../dataset/iris/iris.data')

    pass
