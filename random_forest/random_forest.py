#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from collections import defaultdict
import pandas as pd
import random


class DecisionTree(object):
    """docstring for DecisionTree"""

    def __init__(self, split_index=None, attr_value=None,
                 children=None, label=None):
        self.label = label
        self.split_index = split_index
        self.attr_value = attr_value
        self.children = children if children else []

    def add_child(self, child):
        self.children.append(child)

    def predict(self, x):
        if self.children:
            for child in self.children:
                if x[self.split_index] == child.attr_value:
                    return child.predict(x)
        else:
            return self.label


class RandomForest(object):
    """docstring for RandomForest"""

    def __init__(self, tree_num=10, max_features=2):
        self.trees = []
        self.tree_num = tree_num
        self.max_features = max_features

    def fit(self, data_set, attr_set, label_index):
        size = len(attr_set)
        for k in range(self.tree_num):
            sample_set = self.bootstrap(data_set)
            indexes = [random.randint(0, size - 1)
                       for _ in range(self.max_features)]
            features = attr_set.iloc[indexes]
            indexes.append(label_index)
            sample_slice = sample_set.iloc[:, indexes]
            label_set = data_set.iloc[:, label_index]
            self.trees.append(self.build_tree(sample_slice,
                                              features,
                                              label_set))

    def build_tree(self, data_set, attr_set, label_set):
        label_dict = self.count_attr(label_set)
        if len(label_dict.keys()) == 1:
            return DecisionTree(label=label_dict.keys()[0])
        if attr_set.count(None) == len(attr_set):
            values = label_dict.values()
            max_value_index = values.index(max(values))
            max_label = label_dict.keys()[max_value_index]
            return DecisionTree(label=max_label)
        tree = DecisionTree()
        tree.split_index = self.best_split_attr(data_set, attr_set)
        sub_attrs = attr_set[:]
        sub_attrs[tree.split_index] = None
        attr_dict = self.count_attr(data_set.iloc[:, tree.split_index])
        for key, value in attr_dict.items():
            sub_set = self.sub_set(data_set, tree.split_index, key)
            sub_tree = self.build_tree(sub_set,
                                       sub_attrs,
                                       sub_set.iloc[:, -1])
            sub_tree.attr_value = key
            tree.add_child(sub_tree)
        return tree

    def classifier(self, x):
        pass

    def best_split_attr(self, data_set, attrs):
        gains = [self.info_gain(data_set, i) for i in range(len(attrs))]
        best_attr_index = gains[gains.index(max(gains))]
        return best_attr_index

    def info_gain(self, data_set, attr_index):
        '''
        求数据集的信息增益
        数据集D的信息熵 I(D)=-(P1*log(P1,2)+P1*log(P1,2)+...+Pn*log(Pn,2))
                            (Pi 表示类别i样本数量占所有样本的比例)
        数据集D特征X的信息熵I(D|X)=sum((|Xi|/|D|)*I(Xi))(i=1.2...n)
                                (Xi表示特征X的值为Xi的集合,|S|为数据集S的样本数量)
        信息增益g(D,X)=I(D)-I(D|X) 数据集D特征X的信息增益
        '''
        total = len(data_set)
        if total == 0:
            return 0
        entropy = self.info_entropy(data_set.iloc[:, ])
        attr_dict = self.count_attr(data_set, attr_index)
        attr_entropy_expect = 0  # 特征的信息期望
        for key, value in attr_dict.items():
            p = value * 1.0 / total
            sub_set = self.sub_set(data_set, attr_index, key)
            attr_entropy_expect += p * self.info_entropy(
                sub_set.iloc[:, -1])
        return entropy - attr_entropy_expect

    def info_gain_rate(self, attr_set):
        '''求数据集的信息增益率'''
        entropy = self.entropy(attr_set)
        if abs(entropy - 0.0) < 1e-6:
            return 0.
        return self.info_gain(attr_set) / entropy

    def info_entropy(self, label_set):
        '''信息熵'''
        return self.entropy(label_set)

    def entropy(self, attr_set):
        total = len(attr_set)
        if total == 0:
            return 0.
        count_dict = self.count_attr(attr_set)
        entropy = 0.
        for key, value in count_dict.items():
            p = value * 1.0 / total
            entropy += -p * math.log(p, 2)
        return entropy

    def count_attr(self, attr_set):
        count_dict = defaultdict(int, 0)
        for attr in attr_set:
            count_dict[attr] += 1
        return count_dict

    def sub_set(self, data_set, attr_index, attr_value):
        sub_set = []
        for row in data_set:
            if row[attr_index] == attr_value:
                sub_set.append(row)
        return sub_set

    def bootstrap(self, data_set):
        size = len(data_set)
        indexes = [random.randint(0, size - 1) for _ in range(size)]
        return data_set.iloc[indexes, :]


def preprocess(filename):
    train_set = pd.read_csv(filename)
    return train_set


if __name__ == '__main__':
    train_set = preprocess("../dataset/titanic/train.csv")
    test_set = preprocess("../dataset/titanic/test.csv")
    test_set = test_set.iloc[1:]
    attr_set = train_set[0, :]
    train_set = train_set.iloc[1:, :]
    submission = []
    submission.append(['PassengerId', 'Survived'])
    random_forest = RandomForest(10, 3)
    random_forest.fit(train_set, attr_set, 1)
    count = len(test_set)
    for i in range(count):
        label = random_forest.classifier(test_set[i])
        submission.append([test_set[i][0], label])
    submission_df = pd.DataFrame(data=submission,
                                 columns=['PassengerId', 'Survived'])
    submission_df.to_csv('submission.csv', index=False)
