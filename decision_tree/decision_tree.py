#!/usr/bin/python3
import math
import matplotlib as plt


class decision_tree(object):
    def __init__(self, data_set=[], id_index=-1, label_index=-1):
        self.__label_index = label_index
        self.__attrs = [None, None] + data_set[0][2:]
        self.__train_set = data_set[1:]
        self.__id_index = id_index
        self.__root = {}
        pass

    def build_tree_by_id3(self, data_set, attr_index):
        label_dict = self.count_attr(data_set, self.__label_index)
        if len(label_dict) == 1:
            return label_dict.keys()
        elif self.__attrs.count(None) == len(self.__attrs) - 1:
            max_label = None
            for item in label_dict.items():
                label, value = item
                if max_label is None:
                    max_label = label
                    continue
                if value > label_dict[max_label]:
                    max_label = label
            return max_label
        else:
            tree = {}
            self.__attrs[attr_index] = None
            attr_dict = self.count_attr(data_set, attr_index)
            for item in attr_dict.items():
                key, value = item
                sub_set = self.sub_set_for_attr(data_set, attr_index, key)
                best_attr_index = self.best_attr_index(sub_set)
                tree[key] = self.build_tree_by_id3(sub_set, best_attr_index)
            return tree

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
        entropy = self.info_entropy(data_set)
        attr_dict = self.count_attr(data_set, attr_index)
        attr_entropy = 0
        count = len(data_set)
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
        label_dict = self.count_attr(data_set, self.__label_index)
        gini_imp = 0
        count = len(data_set)
        for item in label_dict.items():
            key, value = item
            p = value * 1.0 / count
            gini_imp += p * p
        return 1 - gini_imp

    def sub_set_for_attr(self, data_set, attr_index, attr_value):
        sub_set = []
        for row in data_set:
            if row[attr_index] = attr_value:
                sub_set.append(row)
        return sub_set

    def count_of_attr(self, data_set, attr_index, attr_value):
        count = 0
        for row in data_set:
            if row[attr_index] = attr_value:
                count += 1
        return count

    def info_entropy(self, data_set):
        '''信息熵'''
        label_dict = self.count_attr(data_set, self.__label_index)
        entropy = 0
        count = len(data_set)
        for item in label_dict.items():
            key, value = item
            p = value * 1.0 / count
            entropy += p * math.log(p, 2)
        return -entropy

    def entropy(self, data_set, attr_index):
        '''条件熵'''
        attr_dict = self.count_attr(data_set, attr_index)
        entropy = 0
        count = len(attr_dict)
        for item in attr_dict.items():
            key, value = item
            p = value * 1.0 / count
            entropy += p * math.log(p, 2)
        return -entropy

    def count_attr(self, data_set, attr_index):
        count_dict = {}
        for row in self.__train_set:
            attr_value = row[attr_index]
            if attr_value in attr_dict:
                count_dict[attr_value] += 1
            else:
                count_dict[attr_value] = 1
        return count_dict


class decision_tree_node(object):
    def __init__(self, attr_label="", children={}):
        self.__children = children
        self.__attr_label = attr_label

    def get_children(self):
        return self.__children

    def get_attr_label(self):
        return self.__attr_label
