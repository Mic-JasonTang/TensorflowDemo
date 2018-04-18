#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 22:42
# @Author  : Jasontang
# @Site    : 
# @File    : conv2d_demo.py
# @ToDo    : 

import tensorflow as tf

# 前两个维度代表了过滤器的尺寸5x5，第三个维度代表当前层的深度3，第四个维度表示过滤器的深度16
filiter_weight = tf.get_variable("weights", [5, 5, 3, 16], initializer=tf.constant_initializer(2))

biases = tf.get_variable("biases", [16], initializer=tf.constant_initializer(0.1))

# 四维矩阵，第一维是一个batch，后三维为结点矩阵
input_ = tf.get_variable("input", [1, 5, 5, 3], initializer=tf.constant_initializer(3))

conv = tf.nn.conv2d(input_, filiter_weight, strides=[1, 1, 1, 1], padding="SAME")

bias = tf.nn.bias_add(conv, biases)

actived_conv = tf.nn.relu(bias)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    filiter_weight = sess.run(filiter_weight)
    print(filiter_weight)
    print("-"*30)
    biases = sess.run(biases)
    print(biases)
    print("-"*30)
    input_ = sess.run(input_)
    print(input_)
    result = sess.run(actived_conv)
    print(result)
