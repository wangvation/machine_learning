#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../dataset/mnist', one_hot=True)
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    W = tf.Variable(tf.zeros([784, 10]), name='Weight')
    b = tf.Variable(tf.zeros([10]), name='bias')
with tf.name_scope('output'):
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')
with tf.name_scope('target'):
    y_ = tf.placeholder(tf.float32, [None, 10])
with tf.variable_scope('cost'):
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), axis=1), name='cross_entropy')
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(
        0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.summary.FileWriter('logs/', sess.graph)
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy,
               feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()
