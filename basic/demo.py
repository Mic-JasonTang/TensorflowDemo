#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 23:05
# @Author  : Jasontang
# @Site    : 
# @File    : demo.py
# @ToDo    : tensorflow基础代码

import tensorflow as tf

# 通过设置种子来
# 这里的shape可以换成元组吗
# 可以， if isinstance(shape, (tuple, list)) and not shape:，源码是这样写的。
w1 = tf.Variable(tf.random_normal(shape=(2,3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3,1), stddev=1, seed=1))

# 声明一个1*2的矩阵
# x = tf.constant([[0.7, 0.9]])
# 使用占位符来代替输入x,并定义3*2的矩阵，输入3次x
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

# 矩阵乘法， 这里是计算向前传播的结果
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 初始化
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

    constant1 = tf.constant(0)
    constant2 = tf.constant(1, shape=[1])
    constant3 = tf.constant(1, shape=[3])
    print(constant1)
    print(constant2)
    print(constant3)
    print(sess.run(constant1))
    print(sess.run(constant2))
    print(sess.run(constant3))