#! /usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../common")
import heap
import time
import math
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from csv_utils import *
mod = SourceModule("""
    #include<math.h>
    #include <cuda_runtime.h>
    __global__ void manhattan_distance(int *sample_set,int *item,int *out)
    {
        int pixel_index =threadIdx.x;
        int sample_index=blockIdx.x;
        int x=abs(sample_set[sample_index*blockDim.x+pixel_index]-item[pixel_index]);
        x=atomicAdd(&out[sample_index],x);
        atomicExch(&out[sample_index],x);
    }
    __global__ void eulidean_distance(int *sample_set,int *item,int *out)
    {
        int pixel_index =threadIdx.x;
        int sample_index=blockIdx.x;
        int x=sample_set[sample_index*blockDim.x+pixel_index]-item[pixel_index];
        x*=x;
        x=atomicAdd(&out[sample_index],x);
        atomicExch(&out[sample_index],x);
    }
    """)
manhattan_distance_gpu = mod.get_function("manhattan_distance")
eulidean_distance_gpu = mod.get_function("eulidean_distance")


class Data(object):

    def __init__(self, label, dist):
        self.label = label
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist

    def __le__(self, other):
        return self.dist <= other.dist

    def __eq__(self, other):
        return self.dist == other.dist

    def __ne__(self, other):
        return self.dist != other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __ge__(self, other):
        return self.dist >= other.dist

    def __str__(self):
        return "[" + self.label + ":" + str(self.dist) + "]"


class KNN(object):

    def __init__(self, sample_set, K):
        self.__sample_set = sample_set
        self.__K = K
        pass

    def eulidean_distance(self, item1, item2):
        manhattan_distance_gpu
        temp = np.sum((item1[1:] - item2[1:])**2)
        return temp

    def manhattan_distance(self, item1, item2):
        manhattan_distance_gpu
        temp = np.sum(np.abs(item1[1:] - item2[1:]))
        return temp

    def __findLabel(self, data_seq):
        count_dict = {}
        for data in data_seq:
            if data.label in count_dict:
                count_dict[data.label] += 1
            else:
                count_dict[data.label] = 1
        result_label = None
        max_count = 0
        for key in count_dict.keys():
            if count_dict[key] > max_count:
                result_label = key
                max_count = count_dict[key]
        return result_label

    def classifier(self, item):
        _heap = heap.Heap(type="max_heap", capacity=self.__K)
        sample_count = len(self.__sample_set)
        dist_set = np.zeros((sample_count, 1), dtype=np.int32)
        # manhattan_distance_gpu(drv.In(self.__sample_set), drv.In(item), drv.InOut(
        #     dist_set), block=(len(item), 1, 1), grid=(sample_count, 1, 1))
        eulidean_distance_gpu(drv.In(self.__sample_set), drv.In(item), drv.InOut(
            dist_set), block=(len(item), 1, 1), grid=(sample_count, 1, 1))
        for i in range(sample_count):
            data = Data(self.__sample_set[i][0], dist_set[i])
            if _heap.size() < self.__K or data < _heap.top():
                _heap.push(data)
        return self.__findLabel(_heap.elements())


def normalize(_list):
    for i in range(1, len(_list)):
        _list[i] = _list[i] > 127 and 1 or 0
    return _list


if __name__ == '__main__':
    print("load data...")
    train_set = load_csv("train.csv")
    test_set = load_csv("test.csv")
    print("init data...")
    # train_set = [list(map(int, x)) for x in train_set[1:]]
    # train_set = [normalize(x) for x in train_set]
    # test_set = [list(map(int, [1] + x)) for x in test_set[1:]]
    # test_set = [normalize(x) for x in test_set]

    # train_set = np.array(train_set, dtype=np.int32)
    # test_set = np.array(test_set, dtype=np.int32)
    # print("knn trainning...")
    # knn = KNN(train_set, 199)
    # print("knn test...")
    # submission = []
    # test_count = len(test_set)
    # submission.append(["ImageId", "Label"])
    # for i in range(test_count):
    #     label = knn.classifier(test_set[i])
    #     submission.append([i + 1, label])
    # save_csv("submission.csv", submission)
