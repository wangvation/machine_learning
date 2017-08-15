#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../common")
import math
import matplotlib as plt
from csv_utils import *


class decision_tree(object):
    def __init__(self, data_set=[], id_index=-1, label_index=-1):
        self.__label_index = label_index
        self.__attrs = data_set[0]
        self.__attrs[id_index] = None
        self.__attrs[label_index] = None
        self.__train_set = data_set[1:]
        self.__id_index = id_index
        self.__root = None
        pass

    def fit(self, mod="id3"):
        if mod == "id3":
            self.__root = self.build_tree_by_id3(self.__train_set)
        elif mod == "c45":
            self.__root = self.build_tree_by_c45()
        elif mod == "c50":
            self.__root = self.build_tree_by_c50()
        elif mod == "cart":
            self.__root = self.build_tree_by_cart()

    def build_tree_by_id3(self, data_set, attr_index=-1):
        label_dict = self.count_attr(data_set, self.__label_index)
        if len(label_dict) == 1:
            return label_dict.keys()
        if self.__attrs.count(None) == len(self.__attrs) - 1:
            max_label = None
            for item in label_dict.items():
                label, value = item
                if max_label is None:
                    max_label = label
                    continue
                if value > label_dict[max_label]:
                    max_label = label
            return max_label
        tree = {}
        if attr_index == -1:
            attr_index = self.best_attr_index(data_set)
        self.__attrs[attr_index] = None
        attr_dict = self.count_attr(data_set, attr_index)
        for item in attr_dict.items():
            key, value = item
            sub_set = self.sub_set_for_attr(data_set, attr_index, key)
            best_attr_index = self.best_attr_index(sub_set)
            tree[key] = self.build_tree_by_id3(sub_set, best_attr_index)
        return {attr_index: tree}

    def best_attr_index(self, data_set):
        best_attr_index = -1
        max_gain = None
        for i in range(len(self.__attrs)):
            if self.__attrs[i] is not None:
                gain = self.info_gain(data_set, i)
                if max_gain is None or gain > max_gain:
                    best_attr_index = i
                    max_gain = gain
        return best_attr_index

    def build_tree_by_c45(self):
        pass

    def build_tree_by_c50(self):
        pass

    def build_tree_by_cart(self):
        pass

    def info_gain(self, data_set, attr_index):
        '''
        求数据集的信息增益
        数据集D的信息熵 I(D)=-(P1*log(P1,2)+P1*log(P1,2)+...+Pn*log(Pn,2))
                            (Pi 表示类别i样本数量占所有样本的比例)
        数据集D特征X的信息熵I(D|X)=sum((|Xi|/|D|)*I(Xi))(i=1.2...n)
                                (Xi表示特征X的值为Xi的集合,|S|为数据集S的样本数量)
        信息增益g(D,X)=I(D)-I(D|X) 数据集D特征X的信息增益
        '''
        count = len(data_set)
        if count == 0:
            return 0
        entropy = self.info_entropy(data_set)
        attr_dict = self.count_attr(data_set, attr_index)
        attr_entropy = 0
        for item in attr_dict.items():
            key, value = item
            p = value * 1.0 / count
            sub_set = self.sub_set_for_attr(data_set, attr_index, key)
            attr_entropy += p * self.info_entropy(sub_set)
        attr_entropy = -attr_entropy
        return entropy - attr_entropy

    def info_gain_ratio(self, data_set, attr_index):
        '''求数据集的信息增益率'''
        return self.info_gain(data_set, attr_index) / self.entropy(data_set, attr_index)

    def gini_impurity(self, data_set):
        '''基尼不纯度'''
        count = len(data_set)
        if count == 0:
            return 0
        label_dict = self.count_attr(data_set, self.__label_index)
        gini_imp = 0
        for item in label_dict.items():
            key, value = item
            p = value * 1.0 / count
            gini_imp += p * p
        return 1 - gini_imp

    def sub_set_for_attr(self, data_set, attr_index, attr_value):
        sub_set = []
        for row in data_set:
            if row[attr_index] == attr_value:
                sub_set.append(row)
        return sub_set

    def count_of_attr(self, data_set, attr_index, attr_value):
        count = 0
        for row in data_set:
            if row[attr_index] == attr_value:
                count += 1
        return count

    def info_entropy(self, data_set):
        count = len(data_set)
        if count == 0:
            return 0
        label_dict = self.count_attr(data_set, self.__label_index)
        entropy = 0
        for item in label_dict.items():
            key, value = item
            p = value * 1.0 / count
            entropy += p * math.log(p, 2)
        return -entropy

    def entropy(self, data_set, attr_index):
        count = len(attr_dict)
        if count == 0:
            return 0
        attr_dict = self.count_attr(data_set, attr_index)
        entropy = 0
        for item in attr_dict.items():
            key, value = item
            p = value * 1.0 / count
            entropy += p * math.log(p, 2)
        return -entropy

    def count_attr(self, data_set, attr_index):
        count_dict = {}
        for row in self.__train_set:
            attr_value = row[attr_index]
            if attr_value in count_dict:
                count_dict[attr_value] += 1
            else:
                count_dict[attr_value] = 1
        return count_dict

    def tree(self):
        return self.__root

    def classifier(self, item):
        node = self.__root
        while isinstance(node, dict):
            key, value = node.items()[0]
            node = value[item[key]]
        return node

    def dump(self):
        pass


if __name__ == '__main__':
    train_set = load_csv("train.csv")
    train_set = [x[:3] + x[4:] for x in train_set]
    test_set = load_csv("test.csv")
    gender_submission = load_csv("gender_submission.csv")
    test_set = test_set[1:]
    test_set = [[x[0]] + [None] + [x[1]] + x[3:] for x in test_set]
    # length = len(train_set)
    # splie_index = length * 3 / 4
    # test_set = train_set[splie_index:]
    decision_tree = decision_tree(train_set, id_index=0, label_index=1)
    decision_tree.fit(mod="id3")
    count = len(test_set)
    right_count = 0
    for i in range(count):
        label = decision_tree.classifier(test_set[i])
        if label == gender_submission[i + 1][1]:
            right_count += 1
    print(str(right_count) + "/" + str(count))
