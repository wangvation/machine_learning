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
        self.kernel_mat = self.kernel_trans(self.data_mat, self.data_mat)
        self.errors = [0 for _ in range(self.data_size)]
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
        if alpha <= L:
            return L
        if alpha >= H:
            return H
        return alpha

    def get_new_bias(self, bias1, bias2, alpha1, alpha2):
        if 0 < alpha1 < self.C:
            return bias1
        if 0 < alpha2 < self.C:
            return bias2
        return (bias1 + bias2) / 2.0

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
            return math.exp(-delta / (2.0 * sigma**2))
        elif func == 'linear':
            return np.dot(x, z.T)

    def kernel_trans(self, X, Z):
        m, _ = np.shape(X)
        k, _ = np.shape(Z)
        kernel_mat = np.zeros((m, k))
        for i in range(m):
            for j in range(k):
                kernel_mat[i, j] = self.kernel(X[i, :], Z[j, :])
        return kernel_mat

    def update_alpha_bias(self, index1, index2):

        alpha1_old = self.alphas[index1]
        alpha2_old = self.alphas[index2]
        label1 = self.label_mat[index1]
        label2 = self.label_mat[index2]
        error1 = self.errors[index1]
        error2 = self.errors[index2]

        eta = self.kernel_mat[index1, index1] +\
            self.kernel_mat[index2, index2] -\
            2.0 * self.kernel_mat[index1, index2]
        if eta <= 0:
            return 0
        alpha2_new = alpha2_old + label2 * (error1 - error2) / eta
        L, H = self.get_bounds(alpha1_old, alpha2_old, label1, label2)
        if L == H:
            return 0
        alpha2_clip = self.clip_alpha(alpha2_new, L, H)
        if abs(alpha2_clip - alpha2_old) < 0.00001:
            return 0
        self.alphas[index2] = alpha2_clip
        alpha1_new = alpha1_old + label1 * \
            label2 * (alpha2_old - alpha2_clip)
        self.alphas[index1] = alpha1_new

        b1_new = self.bias + error1 - label1 * \
            (alpha1_new - alpha1_old) * self.kernel_mat[index1, index1] + \
            label2 * (alpha2_clip - alpha2_old) * \
            self.kernel_mat[index1, index2]

        b2_new = self.bias + error2 - label1 * \
            (alpha1_new - alpha1_old) * self.kernel_mat[index1, index2] + \
            label2 * (alpha2_clip - alpha2_old) * \
            self.kernel_mat[index2, index2]

        self.bias = self.get_new_bias(b1_new, b2_new, alpha1_new, alpha2_new)
        self.update_errors(index1)
        self.update_errors(index2)
        return 1

    def is_meet_KKT(self, index):
        func_dist = float(self.label_mat[index]) * self.predict(index)
        if func_dist <= 1:
            return self.alphas[index] == self.C
        if func_dist == 1:
            return 0 < self.alphas[index] < self.C
        if func_dist >= 1:
            return self.alphas[index] == 0

    def select_second_alpha(self, first_index,
                            valid_indexes, debug=False):
        max_error = 0
        second_index = -1
        error_1 = self.errors[first_index]
        for j in valid_indexes:
            if j is not first_index:
                error_j = self.errors[j]
                error = abs(error_1 - error_j)
                # if debug:
                #     print(error)
                if error >= max_error:
                    max_error = error
                    second_index = j
        return second_index

    def update_errors(self, index):
        error = self.predict(index) - self.label_mat[index]
        self.errors[index] = float(error)

    def smo(self):
        self.bias = 0.0
        self.alphas = np.zeros((self.data_size, 1), dtype=np.float32)
        _iter = 0
        entier_flag = True
        update_num = 0
        for i in range(self.data_size):
            self.update_errors(i)
        debug = True
        while (_iter < self.max_iter and update_num > 0) or entier_flag:
            update_num = 0
            if entier_flag:
                valid_indexes = range(self.data_size)
                for i in valid_indexes:
                    if self.is_meet_KKT(i):
                        continue
                    index2 = self.select_second_alpha(i, valid_indexes, False)
                    if index2 is -1:
                        continue
                    update_num += self.update_alpha_bias(i, index2)
                entier_flag = False
            else:
                valid_indexes = np.nonzero(np.multiply(
                    self.alphas < self.C, self.alphas > 0))[0]
                if debug:
                    debug = False
                    print(valid_indexes)
                for i in valid_indexes:
                    index2 = self.select_second_alpha(i, valid_indexes, True)
                    if index2 is -1:
                        continue
                    update_num += self.update_alpha_bias(i, index2)

            # if _iter % 100 == 0:
            #     print(np.sum(np.array(self.errors)**2))
            _iter += 1
        self.get_support_vec()

    def get_support_vec(self):
        sv_index = np.nonzero(self.alphas > 0)[1]
        self.support_vec = self.data_mat[sv_index]
        return self.support_vec

    def predict(self, index):
        kernels = self.kernel_mat[:, index]
        y = np.dot(np.multiply(self.alphas, self.label_mat).T, kernels) \
            + self.bias
        return float(y)

    def classifier(self, X):
        sv_index = np.nonzero(self.alphas > 0)[1]
        alphas = self.alphas[sv_index]
        labels = self.label_mat[sv_index]
        kernels = self.kernel_trans(self.support_vec, np.mat(X))
        y = np.dot(np.multiply(alphas, labels).T, kernels) + self.bias
        return 1 if y >= 0 else -1


if __name__ == '__main__':
    data = pd.read_csv(
        '../dataset/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    col_list = ['id', 'Clump_Thickness', 'Uniformity_of_Cell_Size',
                'Uniformityof_Cell_Shape', 'Marginal_Adhesion',
                'Single_Epithelial', 'Bare_Nuclei', 'Bland_Chromatin',
                'Normal_Nucleoli', 'Mitoses', 'Class']
    # remove the abnormal data
    data = data[data.Bare_Nuclei != '?']
    data.iloc[:, 6] = data.iloc[:, 6].apply(lambda x: float(x))
    # 0 for benign, 1 for malignant
    data[col_list[-1]] = data[col_list[-1]].apply(lambda x: 1 if x == 4 else -1)
    data_size = len(data.values)
    data_index = range(data_size)
    # k - fold CrossValidation
    errors = 0
    rights = 0
    test_index = [random.randint(0, data_size - 1)
                  for _ in range(data_size / 10)]
    train_index = list(set(data_index) ^ set(test_index))
    test_set = data[col_list[1:-1]].iloc[test_index].values
    test_label = data[col_list[-1]].iloc[test_index].values
    train_set = data[col_list[1:-1]].iloc[train_index].values
    label_set = data[col_list[-1]].iloc[train_index].values
    # classifier = svm(C=100, max_iter=10000, func='rbf', sigma=0.9)
    classifier = svm(C=100, max_iter=10000, func='linear')
    classifier.fit(train_set, label_set)
    for i in range(len(test_label)):
        result_label = classifier.classifier(test_set[i])
        print(result_label)
        if result_label != test_label[i]:
            errors += 1
        else:
            rights += 1
    print([rights, errors])
