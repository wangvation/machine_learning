#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import copy
import scipy.io
import matplotlib.pyplot as plt
import time


class sparse_autocoder(object):
    def __init__(self, layers=[], alpha=0.3, toler=0.1,
                 max_iter=1000, lamda=0.00001,
                 beta=0.1, rho=0.05):
        self.hidden_layers = layers[1]
        self.alpha = alpha
        self.max_iter = max_iter
        self.toler = toler
        self.num_layers = 3
        self.beta = beta
        self.rho = rho
        self.lamda = lamda
        self.weights = []
        self.bias = []

        self.weights.append(np.random.rand(layers[1], layers[0]) - 0.5)
        self.bias.append(np.zeros((self.hidden_layers, 1)))
        self.weights.append(np.random.rand(layers[2], layers[1]) - 0.5)
        self.bias.append(np.zeros((layers[2], 1)))

    def fit(self, train_set, method='BGD', is_debug=False):
        self.data_mat = np.mat(train_set)
        self.train(method, is_debug)

    def forward(self, indexes, weights, bias):
        rho_avg = np.zeros((self.hidden_layers, 1))
        outs = []
        for i in indexes:
            actives = []
            actives.append(self.data_mat[:, i])
            Z_L1 = np.dot(weights[0], actives[0]) + bias[0]
            actives.append(self.sigmoid(Z_L1))
            rho_avg += actives[1]
            Z_L2 = np.dot(weights[1], actives[1]) + bias[1]
            actives.append(self.sigmoid(Z_L2))
            outs.append(actives)
        m = len(indexes)
        rho_avg = rho_avg / m
        return outs, rho_avg

    def train(self, method, is_debug):
        _iter = 0
        converges = False
        while _iter < self.max_iter and not converges:
            if is_debug and _iter == self.max_iter / 10:
                indexes = self.get_batch('SGD')
                self.gradient_check(indexes)
                is_debug = False
            converges = self.gradient_descent(method)
            _iter += 1

    def gradient_descent(self, method):
        weights_delta = [np.zeros(w.shape) for w in self.weights]
        bias_delta = [np.zeros(b.shape) for b in self.bias]
        indexes = self.get_batch(method)
        m = len(indexes)
        outs, rho_avg = self.forward(indexes, self.weights, self.bias)
        errors = 0
        for i, actives in zip(indexes, outs):
            target = self.data_mat[:, i]
            error = self.get_item_error(actives[-1], target)
            errors += error
            if error <= 0.01:
                continue
            weights_grad, bias_grad = self.back_propagation(
                self.weights, self.bias, i, rho_avg, actives)
            for k in range(2):
                weights_delta[k] += weights_grad[k]
                bias_delta[k] += bias_grad[k]
        self.weights[0] -= self.alpha * \
            ((1.0 / m) * weights_delta[0] + self.lamda * self.weights[0])
        self.bias[0] -= self.alpha * (1.0 / m) * bias_delta[0]
        self.weights[1] -= self.alpha * \
            ((1.0 / m) * weights_delta[1] + self.lamda * self.weights[1])
        self.bias[1] -= self.alpha * (1.0 / m) * bias_delta[1]
        # print(errors / m)
        if errors / m < self.toler:
            return True
        return False

    def gradient_check(self, indexes):
        weights = copy.deepcopy(self.weights)
        bias = copy.deepcopy(self.bias)
        outs, rho_avg = self.forward(indexes, self.weights, self.bias)
        for k, actives in zip(indexes, outs):
            weights_grad, bias_grad = self.back_propagation(
                weights, bias, k, rho_avg, actives)
            epsilon = 10e-4
            for layer_index in range(2):
                weight = weights[layer_index]
                m, n = np.shape(weight)
                for i in range(m):
                    for j in range(n):
                        weight[i, j] += epsilon
                        error1 = self.get_item_error(
                            self.predict(self.data_mat[:, k],
                                         weights, bias), self.data_mat[:, k])
                        weight[i, j] -= 2 * epsilon
                        error2 = self.get_item_error(
                            self.predict(self.data_mat[:, k],
                                         weights, bias), self.data_mat[:, k])
                        weight[i, j] += epsilon
                        print(weights_grad[layer_index][i, j],
                              (error1 - error2) / (2 * epsilon))
                print('###############################################' +
                      '################################################')

        pass

    def back_propagation(self, weights, bias, index, rho_avg, out):
        '''
        p^平均活跃度,p稀疏性参数,通常是一个接近于0的较小的值(比如0.05)
        beta 控制稀疏性惩罚因子的权重
                                                       p    1-p    ,
        delta(l)=((\sum W_ji(l)*delta_j(l+1))+beta*(- ——- + ————))f(z_i(l))
                                                       p^   1-p^
        '''
        weights_grad = [None, None]
        bias_grad = [None, None]
        target = self.data_mat[:, index]
        output_index = 2
        hidden_index = 1
        input_index = 0
        delta_output = -np.multiply(target - out[output_index],
                                    self.sigmoid_prime(out[output_index]))
        weights_grad[hidden_index] = np.dot(delta_output, out[hidden_index].T)
        bias_grad[hidden_index] = delta_output
        punish = self.beta * \
            ((1 - self.rho) / (1 - rho_avg) - self.rho / rho_avg)
        # print(punish)
        delta_hidden = np.multiply(
            np.dot(self.weights[hidden_index].T, delta_output) + punish,
            self.sigmoid_prime(out[hidden_index]))
        weights_grad[input_index] = np.dot(delta_hidden, out[input_index].T)
        bias_grad[input_index] = delta_hidden
        return weights_grad, bias_grad

    def get_batch(self, method='BGD'):
        data_size, _ = np.shape(self.data_mat)
        if method == 'BGD':  # Batch gradient descent
            indexes = range(data_size)
        elif method == 'SGD':  # Stochastic gradient descent
            indexes = [random.randint(0, data_size - 1)]
        elif method == 'MBGD':  # Mini-batch gradient descent
            m = 10
            indexes = [random.randint(0, data_size - 1) for x in range(m)]
        return indexes

    def get_item_error(self, out, y):
        minus = out - y
        return np.sum(np.multiply(minus, minus)) / 2.0

    def autocode(self, x):
        Zh = np.dot(self.weights[0], np.mat(x).T) + self.bias[0]
        Ah = self.sigmoid(Zh)
        Zo = np.dot(self.weights[1], Ah) + self.bias[1]
        return self.sigmoid(Zo)
        # return Oi

    def predict(self, x, weights, bias):
        actives = []
        actives.append(x)
        Z_L1 = np.dot(weights[0], actives[0]) + bias[0]
        actives.append(self.sigmoid(Z_L1))
        Z_L2 = np.dot(weights[1], actives[1]) + bias[1]
        actives.append(self.sigmoid(Z_L2))
        return actives[-1]

    def sigmoid_prime(self, y):
        return np.multiply(y, (1 - y))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


def displayImage(images, rows, cols, patch_side):
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True, )
    i = 0
    for axis in ax.flat:
        img = images[:, i].reshape(patch_side, patch_side)
        # img = np.array(images[:, :, i], dtype=np.uint8)
        axis.imshow(img, cmap=plt.cm.gray,
                    interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        i += 1

    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    plt.show()


def normalize_dataset(dataset):
    ''' Normalize the dataset provided as input '''
    # Remove mean of dataset

    dataset = dataset - np.mean(dataset)

    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """

    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev

    """ Rescale from [-1, 1] to [0.1, 0.9] """

    dataset = (dataset + 1) * 0.4 + 0.1

    return dataset


def load_dataset(num_patches, patch_side):
    """ Load images into numpy array """

    images = scipy.io.loadmat('../dataset/autocode/IMAGES.mat')
    images = images['IMAGES']

    """ Initialize dataset as array of zeros """

    dataset = np.zeros((patch_side * patch_side, num_patches))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """

    rand = np.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size=(num_patches, 2))
    image_number = rand.randint(10, size=num_patches)

    """ Sample 'num_patches' random image patches """

    for i in xrange(num_patches):

        """ Initialize indices for patch extraction """

        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        """ Extract patch and store it as a column """

        patch = images[index1:index1 + patch_side,
                       index2:index2 + patch_side, index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    """ Normalize and return the dataset """

    dataset = normalize_dataset(dataset)
    return dataset


if __name__ == '__main__':
    vis_patch_side = 16      # side length of sampled image patches
    hid_patch_side = 10      # side length of representative image patches
    rho = 0.05   # desired average activation of hidden units
    lamda = 0.0001  # weight decay parameter
    beta = 3      # weight of sparsity penalty term
    num_patches = 10000  # number of training examples
    max_iterations = 4000    # number of optimization iterations
    input_size = vis_patch_side * vis_patch_side  # number of input units
    hidden_size = hid_patch_side * hid_patch_side  # number of hidden units

    """ Load randomly sampled image patches as dataset """

    training_data = load_dataset(num_patches, vis_patch_side)
    # print(training_data.shape)
    autocoder = sparse_autocoder(layers=[input_size, hidden_size, input_size],
                                 alpha=0.3, toler=0.1, max_iter=max_iterations,
                                 lamda=lamda, beta=beta, rho=rho)
    autocoder.fit(train_set=training_data, method='MBGD', is_debug=False)
    displayImage(training_data[:, :100], rows=10,
                 cols=10, patch_side=vis_patch_side)
    result = np.zeros((input_size, 100))
    result = np.mat(result)
    for i in range(100):
        result[:, i] = autocoder.autocode(training_data[:, i])
    displayImage(result, rows=10, cols=10, patch_side=vis_patch_side)
