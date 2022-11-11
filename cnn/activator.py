#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def relu(array):
    array[array < 0] = 0.0
    return array


def relu_derive(y):
    derive = np.zeros(y.shape)
    derive[y > 0] = 1.0
    return derive


def softplus(x):
    return np.log(1 + np.exp(x))


def softplus_derive(y):
    exp_y = np.exp(y)
    return exp_y / (exp_y + 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derive(y):
    return np.multiply(y, (1 - y))


def tanh(x):
    return np.tanh(x)


def tanh_derive(y):
    return 1.0 - np.power(y, 2)
