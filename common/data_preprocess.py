#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def min_max_normalize(data):
    _min = np.min(data)
    _max = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i] - _min) / (_max - _min)


def zero_score_nomalize(data):
    _mean = np.mean(data)
    _std = np.std(data)
    for i in range(len(data)):
        data[i] = (data[i] - _mean) / _std


def min_max_normalize_np(data):
    _min = np.min(data)
    _max = np.max(data)
    data = (data - _min) / (_max - _min)
    return data


def zero_score_nomalize_np(data):
    _mean = np.mean(data)
    _std = np.std(data)
    data = (data - _mean) / _std
    return data
