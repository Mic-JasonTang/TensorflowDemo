#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/4 15:09
# @Author  : Jasontang
# @Site    : 
# @File    : learningrate_regularizer.py
# @ToDo    : 学习率和正则化

import tensorflow as tf

# 学习率设置
global_step = tf.Variable(100)
# 通过exponential_decay函数生成学习率
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.9)

# 正则化设置
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
v1 = tf.constant(0)
v2 = tf.Variable(0)  # 变量才需要初始化


with tf.Session() as sess:

    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))

    # 初始化变量
    sess.run(global_step.initializer)
    # 得到学习率
    print(sess.run(learning_rate))

