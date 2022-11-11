#! /usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pandas as pd
sys.path.append("../common")
from heap import Heap


mod = SourceModule("""
    # include<math.h>
    # include <cuda_runtime.h>
    __global__ void manhattan_distance(float *sample_set,float *item,float *out)
    {
        int pixel_index =threadIdx.x;
        int sample_index=blockIdx.x;
        float x=abs(sample_set[sample_index*blockDim.x+pixel_index]
                  -item[pixel_index]);
        x=atomicAdd(&out[sample_index],x);
        atomicExch(&out[sample_index],x);
    }
    __global__ void eulidean_distance(float *sample_set,float *item,float *out)
    {
        int pixel_index =threadIdx.x;
        int sample_index=blockIdx.x;
        float x=sample_set[sample_index*blockDim.x+pixel_index]-
                item[pixel_index];
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

    def __init__(self, train_set, targets, K):
        self.__train_set = train_set
        self.__targets = targets
        self.__K = K
        pass

    def eulidean_distance(self, item1, item2):
        temp = np.sum((item1 - item2)**2)
        return temp

    def manhattan_distance(self, item1, item2):
        temp = np.sum(np.abs(item1 - item2))
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
        _heap = Heap(type="max_heap", capacity=self.__K)
        sample_count = len(self.__train_set)
        dist_set = np.zeros((sample_count, 1), dtype=np.float32)
        eulidean_distance_gpu(drv.In(self.__train_set), drv.In(item),
                              drv.InOut(dist_set), block=(len(item), 1, 1),
                              grid=(sample_count, 1, 1))
        for i in range(sample_count):
            data = Data(self.__train_set[i], dist_set[i])
            if _heap.size() < self.__K or data < _heap.top():
                _heap.push(data)
        return self.__findLabel(_heap.elements())


def normalize(array):
    return array.astype(np.float32) / 255.0


if __name__ == '__main__':
    print("load data...")
    train_data = pd.read_csv("../dataset/digit_recognizer/train.csv")
    test_data = pd.read_csv("../dataset/digit_recognizer/test.csv")
    print("init data...")
    train_set = train_data.iloc[:, 1:].values
    test_set = test_data.values
    train_set = normalize(train_set)
    test_set = normalize(test_set)

    print("knn trainning...")
    knn = KNN(train_set, 20)
    print("knn test...")
    submission = []
    test_count = len(test_set)
    for i in range(test_count):
        label = knn.classifier(test_set[i])
        submission.append([i + 1, label])
    submission_df = pd.DataFrame(data=submission,
                                 columns=['ImageId', 'Label'])
    submission_df.to_csv(
        '../dataset/digit_recognizer/submission.csv', index=False)
