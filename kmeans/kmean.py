#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from collections import Counter
from collections import defaultdict

EPSILON = 1e-4


class KMeans(object):
    """docstring for KMeans"""

    def __init__(self, K):
        self.K = K
        self.cent_mass = None
        self.clusters = defaultdict(list)
        self.label_dict = {}

    def fit(self, train_set, label_set):
        self.data_size = train_set.shape[0]
        self.train_set = train_set
        self.label_set = label_set
        indexes = [random.randint(0, self.data_size)
                   for _ in range(self.K)]
        self.cent_mass = self.train_set[indexes]
        _iter = 0
        while self.cluster() > 0 and self.update_cent_mass() > 0:
            _iter += 1
            _iter % 100 == 0 and print(_iter)
            if _iter > 1000:
                break
            pass
        self.bind_label()

    def cluster(self):
        clusters = [[] for _ in range(self.K)]
        for i, item in enumerate(self.train_set, 0):
            min_dist_square = None
            cm_index = None
            for k, cm in enumerate(self.cent_mass, 0):
                dist_square = self.eulidean_dist_square(item, cm)
                if min_dist_square is None or dist_square < min_dist_square:
                    min_dist_square = dist_square
                    cm_index = k
            clusters[cm_index].append(i)
        update = 0
        for k in range(self.K):
            if not self.cluster_equal(clusters[k], self.clusters[k]):
                self.clusters[k] = clusters[k]
                update += 1
        return update

    def cluster_equal(self, cluster1, cluster2):
        len1 = len(cluster1)
        len2 = len(cluster2)
        if len1 == 0 or len2 == 0 or len1 != len2:
            return False
        for c1, c2 in zip(cluster1, cluster2):
            if c1 != c2:
                return False
        return True

    def eulidean_dist_square(self, item, cent_mass):
        minus = item - cent_mass
        dist_square = np.sum(np.multiply(minus, minus))
        return dist_square

    def update_cent_mass(self):
        update = 0
        for k, cluster in self.clusters.items():
            _len = len(cluster)
            if _len == 0:
                continue
            new_cent_mass = (np.sum(self.train_set[cluster], axis=0) / _len)
            dist_square = self.eulidean_dist_square(new_cent_mass,
                                                    self.cent_mass[k])
            if dist_square > EPSILON:
                self.cent_mass[k] = new_cent_mass
                update += 1
        return update

    def bind_label(self):
        for k, cluster in self.clusters.items():
            # print('cluster:k=%d,len=%d', k, len(cluster))
            conter = Counter(self.label_set[cluster])
            label_count = conter.most_common(1)
            if len(label_count) == 0:
                continue
            label, count = label_count[0]
            self.label_dict[k] = label

    def classifier(self, x):
        min_dist_square = None
        cm_index = None
        for k, cm in enumerate(self.cent_mass, 0):
            dist_square = self.eulidean_dist_square(x, cm)
            if min_dist_square is None or dist_square < min_dist_square:
                min_dist_square = dist_square
                cm_index = k
        return self.label_dict[cm_index]


if __name__ == '__main__':
    data = pd.read_csv(
        '../dataset/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    col_list = ['id', 'Clump_Thickness', 'Uniformity_of_Cell_Size',
                'Uniformityof_Cell_Shape', 'Marginal_Adhesion',
                'Single_Epithelial', 'Bare_Nuclei', 'Bland_Chromatin',
                'Normal_Nucleoli', 'Mitoses', 'Class']
    # remove the abnormal data
    data = data[data.Bare_Nuclei != '?']
    data.iloc[:, 6] = data.iloc[:, 6].apply(lambda x: float(x))
    # 0 for benign, 1 for malignant
    data[col_list[-1]] = data[col_list[-1]].apply(lambda x: 1 if x == 4 else 0)
    data_size = len(data.values)
    # k - fold CrossValidation
    K = 10
    data_index = range(data_size)
    errors = []
    rights = []
    for i in range(K):
        test_index = [random.randint(0, data_size - 1)
                      for _ in range(data_size // K)]
        train_index = list(set(data_index) ^ set(test_index))
        test_set = data[col_list[1:-1]].iloc[test_index].values
        test_label = data[col_list[-1]].iloc[test_index].values
        train_set = data[col_list[1:-1]].iloc[train_index].values
        label_set = data[col_list[-1]].iloc[train_index].values
        classifier = KMeans(2)
        classifier.fit(train_set, label_set)
        result_set = [classifier.classifier(test) for test in test_set]
        err = 0
        right = 0
        for predict, label in zip(result_set, test_label):
            if predict != label:
                err += 1
            else:
                right += 1

        errors.append(err * 1.0 / len(test_set))
        rights.append([right, len(test_set)])
    print(np.mean(errors))
    print(rights)
