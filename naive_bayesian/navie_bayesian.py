#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../common")
import pandas as pd


class NvaieBayesian(object):
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
        self._attrs = train_set[0]
        self._train_set = train_set[1:]
        self._attrs_set = {x: {} for x in self._attrs}
        self._label_set = label_set
        self._label_p = {}
        pass

    def fit(self):
        for label in self._label_set:
            if label in self._label_p:
                self._label_p[label] += 1
            else:
                self._label_p[label] = 1
        train_set_size = len(self._train_set)
        for key, value in self._label_p.items():
            self._label_p[key] = value / train_set_size
        for example in self._train_set:
            example_size = len(example)
            for i in range(example_size):
                attr = example[i]
                attr_p = self._attrs_set[self._attrs[i]]
                if attr in attr_p:
                    attr_p[attr] += 1
                else:
                    attr_p[attr] = 1
        for x in self._attrs:
            attr_p = self._attrs_set[x]
            for k, v in attr_p.items():
                attr_p[k] = v / train_set_size

    def classifier(self, item):
        best_label = None
        best_p = None
        px = 1
        pxory = 1  # P(X|Y)
        for label, py in self._label_p.items():
            for i in range(len(item)):
                attr = self._attrs[i]
                attr_p = self._attrs_set[attr]
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
        for i in range(len(self._train_set)):
            if (self._train_set[i][attr_index] == attr and
                    self._label_set[i] == label):
                num += 1
        if num == 0:
            return 1
        p = num / len(self._train_set)  # p(XiY)
        p = p / (self._label_p[label] + 1)  # p(Xi|Y)=p(XiY)/p(Y)
        return p

    def attr_set(self):
        return self._attrs_set

    def label_p(self):
        return self._label_p


if __name__ == '__main__':
    train_data = pd.read_csv("train.csv")
    col_list = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    train_set = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                            'Fare', 'Cabin', 'Embarked']].values[1:]
    label_set = train_data['Survived'].values[1:]
    debug = True
    if debug:
        length = len(train_set)
        splie_index = length * 3 // 4
        test_set = train_set[splie_index:]
        test_label = label_set[splie_index:]
        bayesian = NvaieBayesian(train_set, label_set)
        bayesian.fit()
        print(bayesian.label_p())
        count = len(test_set)
        right_count = 0
        for i in range(count):
            label = bayesian.classifier(test_set[i])
            if label == test_label[i]:
                right_count += 1
        print(str(right_count) + "/" + str(count))
    else:
        test_data = pd.read_csv("test.csv")
        test_set = test_data[['Pclass', 'Sex', 'Age', 'SibSp',
                              'Parch', 'Ticket', 'Fare', 'Cabin',
                              'Embarked']].values[1:]
        bayesian = NvaieBayesian(train_set, label_set)
        bayesian.fit()
        print(bayesian.label_p())
        submission = []
        submission.append(['PassengerId', 'Survived'])
        count = len(test_set)
        for i in range(count):
            label = bayesian.classifier(test_set[i])
            submission.append([test_set[i][0], label])
        submission_df = pd.DataFrame(data=submission,
                                     columns=['PassengerId', 'Survived'])
        submission_df.to_csv('submission.csv', index=False)
