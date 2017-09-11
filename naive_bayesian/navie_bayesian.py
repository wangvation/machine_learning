#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../common")
from csv_utils import *


class nvaie_bayesian(object):
    '''
    朴素贝叶斯分类器
    条件概率公式：P(Y|X)=P(XY)/P(X) (X特征条件，Y类别标签)
    乘法公式：P(XY)=P(Y|X)P(X)
    贝叶斯公式：P(Y|X)=P(X|Y)*P(Y)/P(X)
    假设特征X的维度为n,取值X1,X2,···,Xn,且假设特征之间相互独立(之所以叫朴素贝叶斯就是这个原因)
    则P(X)=P(X1)*P(X2)*···*P(Xn),P(X|Y)=P(X1|Y)*P(X2|Y)*···*P(Xn|Y)
    所以P(Y|X)=[P(X1|Y)*P(X2|Y)*···*P(Xn|Y)*P(Y)]/[P(X1)*P(X2)*···*P(Xn)]
    '''

    def __init__(self, train_set, label_set):
        '''
        '''
        self.__attrs = train_set[0]
        self.__train_set = train_set[1:]
        self.__attrs_set = {x: {} for x in self.__attrs}
        self.__label_set = label_set
        self.__label_p = {}
        pass

    def fit(self):
        for label in self.__label_set:
            if label in self.__label_p:
                self.__label_p[label] += 1
            else:
                self.__label_p[label] = 1
        train_set_size = len(self.__train_set)
        for key, value in self.__label_p.items():
            self.__label_p[key] = value / train_set_size
        for example in self.__train_set:
            example_size = len(example)
            for i in range(example_size):
                attr = example[i]
                attr_p = self.__attrs_set[self.__attrs[i]]
                if attr in attr_p:
                    attr_p[attr] += 1
                else:
                    attr_p[attr] = 1
        attrs_size = len(self.__attrs)
        for x in self.__attrs:
            attr_p = self.__attrs_set[x]
            for k, v in attr_p.items():
                attr_p[k] = v / train_set_size

    def classifier(self, item):
        best_label = None
        best_p = None
        px = 1
        pxory = 1  # P(X|Y)
        for label, py in self.__label_p.items():
            for i in range(len(item)):
                attr = self.__attrs[i]
                attr_p = self.__attrs_set[attr]
                if item[i] in attr_p:
                    px *= attr_p[item[i]]
                pxory *= self.condiction_p(label, i, item[i])
            p = pxory * py / (px + 1)
            if best_p is None or best_p < p:
                best_p = p
                best_label = label
        return best_label

    def condiction_p(self, label, attr_index, attr):
        num = 0
        for i in range(len(self.__train_set)):
            if self.__train_set[i][attr_index] == attr and self.__label_set[i] == label:
                num += 1
        if num == 0:
            return 1
        p = num / len(self.__train_set)  # p(XiY)
        p = p / (self.__label_p[label] + 1)  # p(Xi|Y)=p(XiY)/p(Y)
        return p

    def attr_set(self):
        return self.__attrs_set

    def label_p(self):
        return self.__label_p


if __name__ == '__main__':
    train_set = load_csv("train.csv")
    for item in train_set:
        del(item[3])
    label_set = [x[1] for x in train_set[1:]]
    test_set = load_csv("test.csv")[1:]
    for item in test_set:
        del(item[2])
    gender_submission = load_csv("gender_submission.csv")[1:]
    # length = len(train_set)
    # splie_index = length * 3 // 4
    # test_set = [x[1:] for x in train_set[splie_index:]]
    train_set = [x[2:] for x in train_set]
    bayesian = nvaie_bayesian(train_set, label_set)
    bayesian.fit()
    # print(bayesian.attr_set())
    # print("************************************************************************")
    print(bayesian.label_p())
    count = len(test_set)
    right_count = 0
    for i in range(count):
        label = bayesian.classifier(test_set[i][1:])
        if label == gender_submission[i][1]:
            right_count += 1
    print(str(right_count) + "/" + str(count))
