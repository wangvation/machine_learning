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
        for child in self.children:
            if x[self.split_index] == child.attr_value:
                return child.predict(x)
        else:
            return self.label

    def dump(self, spaces=''):
        print(spaces, '{')
        print(spaces, '    ', 'attr_value:', self.attr_value)
        print(spaces, '    ', 'split_index:', self.split_index)
        if self.children:
            print(spaces, '    ', 'children:[')
            for child in self.children:
                child.dump(spaces + '    ')
            print(spaces, '    ', ']')
        else:
            print(spaces, '    ', 'children', None)
        print(spaces, '    ', 'label', self.label)
        print(spaces, '}')

    def __str__(self):
        _dict = {}
        _dict['    attr_value'] = self.attr_value
        _dict['    split_index'] = self.split_index
        if self.children:
            children = []
            for child in self.children:
                children.append(str(child))
            _dict['   children'] = children
        else:
            _dict['   children'] = None
        _dict['    label'] = self.label
        return str(_dict)


class RandomForest(object):
    """docstring for RandomForest"""

    def __init__(self, tree_num=10, max_features=2):
        self.trees = []
        self.tree_num = tree_num
        self.max_features = max_features

    def fit(self, data_set, attr_set, label_set):
        attr_size = len(attr_set)
        for k in range(self.tree_num):
            sample_set, sample_label_set = self.bootstrap(data_set, label_set)
            attr_indexes = []
            while len(attr_indexes) < self.max_features:
                rand_index = random.randint(0, attr_size - 1)
                if attr_indexes.count(rand_index) == 0:
                    attr_indexes.append(rand_index)
            # features = [attr_set[i] for i in attr_indexes]
            self.trees.append(self.build_tree(sample_set,
                                              attr_indexes,
                                              sample_label_set))
            # print(self.trees[-1])

    def build_tree(self, data_set, attr_indexes, label_set):
        label_dict = self.count_value(label_set)
        if len(label_dict.keys()) == 1:
            return DecisionTree(label=list(label_dict.keys())[0])
        if attr_indexes.count(None) == len(attr_indexes):
            max_label = None
            max_num = 0
            for label, num in label_dict.items():
                if num > max_num:
                    max_num = num
                    max_label = label
            return DecisionTree(label=max_label)
        tree = DecisionTree()
        tree.split_index = self.best_split_attr(data_set,
                                                label_set,
                                                attr_indexes)
        sub_attr_indexes = attr_indexes[:]
        sub_attr_indexes[sub_attr_indexes.index(tree.split_index)] = None
        attr_dict = self.count_value(data_set[:, tree.split_index])
        for attr_value, num in attr_dict.items():
            sub_set, sub_label_set = self.sub_set(data_set,
                                                  label_set,
                                                  tree.split_index,
                                                  attr_value)
            sub_tree = self.build_tree(sub_set,
                                       sub_attr_indexes,
                                       sub_label_set)
            sub_tree.attr_value = attr_value
            tree.add_child(sub_tree)
        return tree

    def classifier(self, x):
        classes = [tree.predict(x) for tree in self.trees]
        class_dict = self.count_value(classes)
        max_label = None
        max_vote = 0
        for label, vote in class_dict.items():
            if vote > max_vote:
                max_vote = vote
                max_label = label
        return int(max_label) if max_label is not None else 0

    def best_split_attr(self, data_set, label_set, attr_indexes):
        max_gains = float('-inf')
        best_attr_index = None
        for index in attr_indexes:
            if index is not None:
                gains = self.info_gain(data_set, label_set, index)
                if gains > max_gains:
                    max_gains = gains
                    best_attr_index = index
        return best_attr_index

    def info_gain(self, data_set, label_set, attr_index):
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
        entropy = self.entropy(label_set)
        attr_dict = self.count_value(data_set[:, attr_index])
        attr_entropy_expect = 0  # 特征的信息期望
        for key, value in attr_dict.items():
            p = value * 1.0 / total
            sub_set, sub_label_set = self.sub_set(data_set, label_set,
                                                  attr_index, key)

            attr_entropy_expect += p * self.entropy(sub_label_set)
        return entropy - attr_entropy_expect

    def info_gain_rate(self, data_set, label_set, attr_index):
        '''求数据集的信息增益率'''
        entropy = self.entropy(label_set)
        if abs(entropy - 0.0) < 1e-6:
            return 0.
        return self.info_gain(data_set[:, attr_index],
                              label_set, attr_index) / entropy

    def entropy(self, value_set):
        total = len(value_set)
        if total == 0:
            return 0.
        count_dict = self.count_value(value_set)
        entropy = 0.
        for key, value in count_dict.items():
            p = value * 1.0 / total
            entropy += p * math.log(p, 2)
        return -entropy

    def count_value(self, value_set):
        count_dict = defaultdict(int)
        for attr in value_set:
            # print('value:', attr)
            count_dict[attr] += 1
        return count_dict

    def sub_set(self, data_set, label_set, attr_index, attr_value):
        sub_set = []
        if label_set:
            sub_label_set = []
            for i, row in enumerate(data_set):
                if row[attr_index] == attr_value:
                    sub_set.append(i)
                    sub_label_set.append(label_set[i])
        else:
            sub_label_set = None
            for i, row in enumerate(data_set):
                if row[attr_index] == attr_value:
                    sub_set.append(i)
        return data_set[sub_set, :], sub_label_set

    def bootstrap(self, data_set, label_set):
        size = len(data_set)
        indexes = [random.randint(0, size - 1) for _ in range(size)]
        sample_label_set = [label_set[i] for i in indexes]
        return data_set[indexes, :], sample_label_set


def preprocess(filename):
    train_set = pd.read_csv(filename)
    return train_set


if __name__ == '__main__':
    train_set = preprocess("train.csv")
    test_set = preprocess("test.csv")
    test_set = test_set[['PassengerId', 'Pclass', 'Sex',
                         'Age', 'SibSp', 'Parch']].values
    attr_set = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    label_set = train_set['Survived'].values
    train_set = train_set[['Pclass', 'Sex',
                           'Age', 'SibSp', 'Parch']].values
    submission = []
    random_forest = RandomForest(10, 3)
    random_forest.fit(train_set, attr_set, label_set)
    count = len(test_set)
    for i in range(count):
        label = random_forest.classifier(test_set[i, 1:])
        submission.append([test_set[i, 0], label])
    submission_df = pd.DataFrame(data=submission,
                                 columns=['PassengerId', 'Survived'])
    submission_df.to_csv('submission.csv', index=False)
