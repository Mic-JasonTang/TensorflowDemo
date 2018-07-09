#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/19 21:49
# @Author  : Jasontang
# @Site    : 
# @File    : Conv-Demo.py
# @ToDo    : 

import tensorflow as tf

input = tf.get_variable("input", [5, 5, 1, 3], initializer=tf.random_normal_initializer(seed=1))
weight = tf.get_variable("weight", [3, 3, 3, 2], initializer=tf.random_normal_initializer(seed=1))
bias = tf.get_variable("bias", [2], initializer=tf.constant_initializer(1))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print("input")
    print(sess.run(input))
    print("weight")
    print(sess.run(weight))
    print("bias")
    print(sess.run(bias))
    result = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding="SAME"), bias)
    print("result")
    print(result)
    print(sess.run(result))
