#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
EPSINON = 10e-6


class lenet_5(object):
    """docstring for lenet_5"""

    def __init__(self, arg):
        self.arg = arg

    def relu(self, x):
        return np.max(0, x)

    def relu_prime(self, y):
        return 1.0 if y > EPSINON else 0.0
