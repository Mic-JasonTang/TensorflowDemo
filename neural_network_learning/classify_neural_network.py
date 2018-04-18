#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/30 21:43
# @Author  : Jasontang
# @Site    : 
# @File    : classify_neural_network.py
# @ToDo    : 使用一个简单的神经网络来完成二分

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None可以方便使用不大的batch的大小。
# 在训练时需要把数据分成比较小的bath，但是在测试时，可以一次性
# 使用全部的数据。
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 定义神经网络向前传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 通过随机数生成一个模拟数据集
# 给定一个种子，使得每次随机的结果一致
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
print(X)
Y = [[int(x1 + x2) < 1] for (x1, x2) in X]
print(Y)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            # 随着训练的进行，交叉熵是逐渐变小的，交叉熵越小说明预测的记过和真是的记过差距越小
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
