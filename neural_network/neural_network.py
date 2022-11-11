#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import math
import copy


class NeuralNetwork(object):
  """docstring for NeuralNetwork"""

  def __init__(self,
               layers=[],
               alpha=0.3,
               toler=0.1,
               max_iter=10000,
               lamda=0.0,
               active_func='sigmoid',
               method='BGD',
               loss='MSE'):
    self.layers = layers
    self.alpha = alpha
    self.toler = toler
    self.max_iter = max_iter
    self.lamda = lamda
    if active_func == 'sigmoid':
      self.active = self.sigmoid
      self.active_derive = self.sigmoid_derive
    else:
      self.active = self.tanh
      self.active_derive = self.tanh_derive
    self.method = method
    self.item_loss = (self.mean_square_error
                      if loss == 'MSE' else self.cross_entropy)
    self.num_layers = len(layers)
    self.weights = []
    self.bias = []
    for i in range(1, self.num_layers):
      n = self.layers[i - 1]
      m = self.layers[i]
      self.weights.append(
          np.random.normal(loc=0.0, scale=1.0 / np.sqrt(n), size=(m, n)))
      self.bias.append(np.zeros((m, 1)))

  def fit(self, data_set, target_set, debug=False):
    self.data_mat = np.mat(data_set)
    self.target_mat = np.mat(target_set)
    self.train(debug)

  def train(self, debug):
    data_size, _ = np.shape(self.data_mat)
    _iter = 0
    while _iter < self.max_iter:
      indexes = self.get_batch()
      outs = self.forward(indexes, self.weights, self.bias)
      targets = [self.target_mat[i].T for i in indexes]
      predict = [out[-1] for out in outs]
      batch_size = len(indexes)

      if debug:
        self.gradient_check(outs, targets, indexes)
        debug = False

      errors = self.loss(batch_size, predict, targets)

      if (_iter) % 100 == 0:
        self.alpha = max(1e-4, self.alpha * 0.98)
        print("loss:", errors, " lr:", self.alpha)

      if errors < self.toler:
        break
      self.gradient_descent(outs, targets, batch_size)
      _iter += 1

  def get_batch(self, batch_size=16):
    data_size, _ = np.shape(self.data_mat)
    if self.method == 'BGD':  # Batch gradient descent
      indexes = range(data_size)
    elif self.method == 'SGD':  # Stochastic gradient descent
      indexes = [random.randint(0, data_size - 1)]
    elif self.method == 'MBGD':  # Mini-batch gradient descent
      indexes = [random.randint(0, data_size - 1) for x in range(batch_size)]
    return indexes

  def forward(self, indexes, weights, bias):
    outs = []
    for i in indexes:
      actives = []
      actives.append(self.data_mat[i].T)

      for j in range(1, self.num_layers):
        Oj = np.dot(weights[j - 1], actives[j - 1]) + bias[j - 1]
        active_out = self.active(Oj)
        actives.append(active_out)

      outs.append(actives)
    return outs

  def gradient_descent(self, outs, targets, batch_size):
    weights_delta = [np.zeros(w.shape) for w in self.weights]
    bias_delta = [np.zeros(b.shape) for b in self.bias]

    for target, out in zip(targets, outs):
      error = self.item_loss(out[-1], target)
      if error <= 0.01:
        continue

      weights_grad, bias_grad = self.back_propagation(out, target)
      for k in range(self.num_layers - 1):
        weights_delta[k] += weights_grad[k]
        bias_delta[k] += bias_grad[k]

    for k in range(self.num_layers - 1):
      self.weights[k] -= self.alpha * \
          (weights_delta[k] / batch_size + self.lamda * self.weights[k])
      self.bias[k] -= self.alpha * bias_delta[k] / batch_size

  def back_propagation(self, out, target):
    output_layer = self.num_layers - 1
    if self.item_loss == self.cross_entropy:
      delta = out[output_layer] - target
    else:
      delta = -np.multiply(target - out[output_layer],
                           self.active_derive(out[output_layer]))
    layer = output_layer - 1
    weights_grad = [None for _ in range(self.num_layers - 1)]
    bias_grad = [None for _ in range(self.num_layers - 1)]
    while layer >= 0:
      w_grad = np.dot(delta, out[layer].T)
      weights_grad[layer] = w_grad
      bias_grad[layer] = delta
      delta = np.multiply(np.dot(self.weights[layer].T, delta),
                          self.active_derive(out[layer]))
      layer -= 1

    return weights_grad, bias_grad

  def gradient_check(self, outs, targets, indexes):
    weights = copy.deepcopy(self.weights)
    bias = copy.deepcopy(self.bias)
    target, out, k = targets[0], outs[0], indexes[0]
    weights_grad, bias_grad = self.back_propagation(out, target)
    epsilon = 10e-4
    for layer_index in range(1, self.num_layers):
      weight = weights[layer_index - 1]
      m, n = np.shape(weight)
      for i in range(m):
        for j in range(n):
          weight[i, j] += epsilon
          error1 = self.item_loss(self.predict(self.data_mat[k], weights, bias),
                                  target)
          weight[i, j] -= 2 * epsilon
          error2 = self.item_loss(self.predict(self.data_mat[k], weights, bias),
                                  target)
          weight[i, j] += epsilon
          print("grad check, back-propagation-grad:",
                weights_grad[layer_index - 1][i, j], " true-grad:",
                (error1 - error2) / (2 * epsilon))

  def loss(self, data_size, outs, targets):
    errors = 0
    for out, y in zip(outs, targets):
      ce_loss = self.item_loss(out, y)
      errors += ce_loss

    errors = errors / data_size
    weight_decay = 0
    for weight in self.weights:
      weight_decay += np.sum(np.multiply(weight, weight))
    weight_decay = weight_decay * self.lamda / 2

    errors = errors + weight_decay
    return errors

  def predict(self, x, weights, bias):
    out = np.mat(x).T
    for i in range(1, self.num_layers):
      out = np.dot(weights[i - 1], out) + bias[i - 1]
      out = self.active(out)
    return out

  def classifier(self, x):
    out = self.predict(x, self.weights, self.bias)
    return np.argmax(out)

  def cross_entropy(self, out, y):
    ce_loss = np.sum(-np.multiply(y, np.log(out)) -
                     np.multiply(1 - y, np.log(1 - out)))

    return ce_loss

  def mean_square_error(self, out, y):
    minus = out - y
    return np.sum(np.multiply(minus, minus)) / 2.0

  def softmax(self, x):
    _sum = np.sum(x)
    return x / _sum

  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def sigmoid_derive(self, y):
    return np.multiply(y, (1 - y))

  def tanh(self, x):
    return np.tanh(x)

  def tanh_derive(self, y):
    return 1 - np.multiply(y, y)


def load_dataset(csv_file, has_label):
  data = pd.read_csv(csv_file)
  data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: x / 255.0)
  data_size, columns_size = data.shape
  if has_label:
    data_set = data.iloc[:, 1:].values
    label_set = pd.get_dummies(data['label']).values
  else:
    data_set = data.iloc[:, :].values
    label_set = None

  all_mask = np.repeat(True, data_size)
  for i in range(data_size):
    if not np.isfinite(data_set[i, :]).all():
      all_mask[i] = False

  data_set = data_set[all_mask, :]
  if has_label:
    label_set = label_set[all_mask, :]

  return data_set, label_set


if __name__ == '__main__':
  train_set, train_label = load_dataset("../dataset/digit_recognizer/train.csv",
                                        True)
  data_size, columns_size = train_set.shape
  train_label = np.clip(train_label, 0.05, 0.95)

  test_set, test_label = load_dataset("../dataset/digit_recognizer/test.csv",
                                      False)
  test_count = len(test_set)

  nn = NeuralNetwork(layers=[784, 128, 10],
                     alpha=1.0,
                     toler=0.01,
                     max_iter=10000,
                     lamda=0.0001,
                     active_func='sigmoid',
                     method='MBGD',
                     loss='cross_entropy')
  nn.fit(train_set, train_label, False)

  if test_label:
    right_count = 0
    for i in range(test_count):
      label = nn.classifier(test_set[i])
      if 1 == test_label[i, label]:
        right_count += 1

    print("testset acc:{}, right:{} / total:{}", right_count / test_count,
          right_count, test_count)
  else:
    submission = []
    for i in range(test_count):
      label = nn.classifier(test_set[i])
      submission.append([i + 1, label])
    submission_df = pd.DataFrame(data=submission, columns=['ImageId', 'Label'])
    submission_df.to_csv('../dataset/digit_recognizer/submission.csv',
                         index=False)
