#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
# import matplotlib as plt
import treePlotter as tp
import pandas as pd
from collections import defaultdict


class DecisionTree(object):
    def __init__(self, data_set=[], id_index=-1,
                 label_index=-1, algorithm='id3'):
        self._label_index = label_index
        self._attrs = data_set[0]
        if id_index != -1:
            self._attrs[id_index] = None
        if label_index != -1:
            self._attrs[label_index] = None
        self._train_set = data_set[1:]
        self._algorithm = algorithm
        self._root = None
        pass

    def fit(self):
        if self._algorithm == "id3":
            self._root = self.build_tree_by_id3(self._train_set,
                                                self._attrs)
        elif self._algorithm == "c45":
            self._root = self.build_tree_by_c45(self._train_set,
                                                self._attrs)
        elif self._algorithm == "c50":
            self._root = self.build_tree_by_c50(self._train_set,
                                                self._attrs)
        elif self._algorithm == "cart":
            self._root = self.build_tree_by_cart(self._train_set,
                                                 self._attrs)

    def build_tree_by_id3(self, data_set, attrs):
        label_dict = self.count_attr(data_set, self._label_index)
        if len(label_dict.keys()) == 1:
            return label_dict.keys()[0]
        if attrs.count(None) == len(attrs):
            max_label = None
            for label, value in label_dict.items():
                if max_label is None:
                    max_label = label
                    continue
                if max_label is None or value > label_dict[max_label]:
                    max_label = label
            return max_label
        sub_tree = {}
        attr_index = self.best_split_attr(data_set, attrs)
        sub_attrs = attrs[:]
        sub_attrs[attr_index] = None
        attr_dict = self.count_attr(data_set, attr_index)
        for key, value in attr_dict.items():
            sub_set = self.sub_set(data_set, attr_index, key)
            sub_tree[key] = self.build_tree_by_id3(sub_set, sub_attrs)
        return {attr_index: sub_tree}

    def best_split_attr(self, data_set, attrs):
        best_attr_index = -1
        max_gain = None
        for i in range(len(attrs)):
            if attrs[i] is not None:
                gain = 0
                if self._algorithm == "id3":
                    gain = self.info_gain(data_set, i)
                elif self._algorithm == "c45":
                    gain = self.info_gain_rate(data_set, i)
                elif self._algorithm == "c50":
                    self._root = self.build_tree_by_c50()
                elif self._algorithm == "cart":
                    self._root = self.build_tree_by_cart()
                if max_gain is None or gain > max_gain:
                    best_attr_index = i
                    max_gain = gain
        return best_attr_index

    def build_tree_by_c45(self, data_set, attrs):
        label_dict = self.count_attr(data_set, self._label_index)
        if len(label_dict.keys()) == 1:
            return label_dict.keys()[0]
        if attrs.count(None) == len(attrs):
            max_label = None
            for label, value in label_dict.items():
                if max_label is None:
                    max_label = label
                    continue
                if max_label is None or value > label_dict[max_label]:
                    max_label = label
            return max_label
        sub_tree = {}
        attr_index = self.best_split_attr(data_set, attrs)
        sub_attrs = attrs[:]
        sub_attrs[attr_index] = None
        attr_dict = self.count_attr(data_set, attr_index)
        for key, value in attr_dict.items():
            sub_set = self.sub_set(data_set, attr_index, key)
            sub_tree[key] = self.build_tree_by_c45(sub_set, sub_attrs)
        return {attr_index: sub_tree}

    def build_tree_by_c50(self, data_set, attrs):
        label_dict = self.count_attr(data_set, self._label_index)
        if len(label_dict) == 1:
            return label_dict.keys()
        if self._attrs.count(None) == len(self._attrs) - 1:
            max_label = None
            for label, value in label_dict.items():
                if max_label is None:
                    max_label = label
                    continue
                if value > label_dict[max_label]:
                    max_label = label
            return max_label
        sub_tree = {}
        attr_index = self.best_split_attr(data_set, attrs)
        sub_attrs = attrs[:]
        sub_attrs[attr_index] = None
        attr_dict = self.count_attr(data_set, attr_index)
        for key, value in attr_dict.items():
            sub_set = self.sub_set(data_set, attr_index, key)
            sub_tree[key] = self.build_tree_by_c50(sub_set, sub_attrs)
        return {attr_index: sub_tree}

    def build_tree_by_cart(self, data_set, attrs):
        label_dict = self.count_attr(data_set, self._label_index)
        if len(label_dict) == 1:
            return label_dict.keys()[0]
        if self._attrs.count(None) == len(self._attrs) - 1:
            max_label = None
            for label, value in label_dict.items():
                if max_label is None:
                    max_label = label
                    continue
                if value > label_dict[max_label]:
                    max_label = label
            return max_label
        sub_tree = {}
        attr_index = self.best_split_attr(data_set, attrs)
        sub_attrs = attrs[:]
        sub_attrs[attr_index] = None
        attr_dict = self.count_attr(data_set, attr_index)
        for key, value in attr_dict.items():
            sub_set = self.sub_set(data_set, attr_index, key)
            sub_tree[key] = self.build_tree_by_cart(sub_set, sub_attrs)
        return {attr_index: sub_tree}

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
        entropy = self.info_entropy(data_set)
        attr_dict = self.count_attr(data_set, attr_index)
        attr_entropy_exp = 0  # 特征的信息期望
        for key, value in attr_dict.items():
            p = value * 1.0 / total
            sub_set = self.sub_set(data_set, attr_index, key)
            attr_entropy_exp += p * self.info_entropy(sub_set)
        return entropy - attr_entropy_exp

    def info_gain_rate(self, data_set, attr_index):
        '''求数据集的信息增益率'''
        entropy = self.entropy(data_set, attr_index)
        if abs(entropy - 0.0) < 10e-6:
            return 0
        return self.info_gain(data_set, attr_index) / entropy

    def gini_impurity(self, data_set):
        '''基尼不纯度'''
        total = len(data_set)
        if total == 0:
            return 0
        label_dict = self.count_attr(data_set, self._label_index)
        gini_imp = 0
        for key, value in label_dict.items():
            p = value * 1.0 / total
            gini_imp += p * p
        return 1 - gini_imp

    def sub_set(self, data_set, attr_index, attr_value):
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
        total = len(data_set)
        if total == 0:
            return 0
        label_dict = self.count_attr(data_set, self._label_index)
        entropy = 0
        for key, value in label_dict.items():
            p = value * 1.0 / total
            entropy += -p * math.log(p, 2)
        return entropy

    def entropy(self, data_set, attr_index):
        total = len(data_set)
        if total == 0:
            return 0
        attr_dict = self.count_attr(data_set, attr_index)
        entropy = 0
        for key, value in attr_dict.items():
            p = value * 1.0 / total
            entropy += -p * math.log(p, 2)
        return entropy

    def count_attr(self, data_set, attr_index):
        count_dict = defaultdict(int)
        for row in data_set:
            attr_value = row[attr_index]
            count_dict[attr_value] += 1
        return count_dict

    def tree(self):
        return self._root

    def count_tree(self, tree, label_dict={}):
        if isinstance(tree, dict):
            root_key, root_tree = tree.items()[0]
            for sub_key, sub_tree in root_tree.items():
                if isinstance(sub_tree, dict):
                    self.count_tree(sub_tree, label_dict)
                else:
                    if sub_tree in label_dict:
                        label_dict[sub_tree] += 1
                    else:
                        label_dict[sub_tree] = 1

    def classifier(self, item, tree=None):
        if tree is None:
            tree = self._root
        for root_key, root_tree in tree.items():
            if item[root_key] in root_tree:
                sub_tree = root_tree[item[root_key]]
                if isinstance(sub_tree, dict):
                    return self.classifier(item, sub_tree)
                return sub_tree
            else:  # 出现训练集中没有覆盖到的样本
                label_dict = {}
                self.count_tree(tree, label_dict)
                max_label = None
                max_value = None
                for key, value in label_dict.items():
                    if max_value is None or value > max_value:
                        max_value = value
                        max_label = key
                return max_label

    def loss_function(self, tree, alpha=1):
        leave_num, leaves = self.get_leaves(tree)
        label_dict = {}
        for leave in leaves:
            if leave in label_dict:
                label_dict[leave] += 1
            else:
                label_dict[leave] = 1
        loss = 0
        for label, num in label_dict.items():
            p = num / leave_num
            loss -= num * math.log(p, 2)
        return loss + alpha * self.node_count(self._root)

    def node_count(self, tree):
        attr_index, node = tree.items()[0]
        if not isinstance(node, dict):
            return 1
        count = 1
        for key, sub_tree in node.items():
            node_count, leaves = self.node_count(sub_tree)
            count += node_count
        return count

    def get_leaves(self, tree):
        attr_index, node = tree.items()[0]
        if not isinstance(node, dict):
            return 1, [node]
        leaves_count = 0
        for key, sub_tree in node.items():
            leaves_num, leaves = self.get_leaves(sub_tree)
            leaves_count += leaves_num
            leaves += leaves
        return leaves_count, leaves


if __name__ == '__main__':
    train_set = pd.read_csv("train.csv").values
    test_set = pd.read_csv("test.csv").values
    gender_submission = pd.read_csv("gender_submission.csv").values
    test_set = test_set[1:]
    decision_tree = DecisionTree(
        train_set, id_index=0, label_index=1, algorithm='c45')
    decision_tree.fit()
    debug = False
    if debug:
        tp.createPlot(decision_tree.tree())
    submission = []
    submission.append(['PassengerId', 'Survived'])
    count = len(test_set)
    for i in range(count):
        label = decision_tree.classifier(test_set[i])
        submission.append([test_set[i][0], label])
    submission_df = pd.DataFrame(data=submission,
                                 columns=['PassengerId', 'Survived'])
    submission_df.to_csv('submission.csv', index=False)
