#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/17 20:11
# @Author  : Jasontang
# @Site    : 
# @File    : saver_demo.py
# @ToDo    : 持久化——存储模型


import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
# 声明tf.train.Saver()类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(result))
    saver.save(sess, "./model_saver_demo/saver_demo.ckpt")

