#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 20:26
# @Author  : Jasontang
# @Site    : 
# @File    : hello.py
# @ToDo    : 


import tensorflow as tf

with tf.device("/cpu:0"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='b')
with tf.device("/gpu:1"):
    c = a + b
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# Runs the op.
sess.run(tf.global_variables_initializer())
print(sess.run(c))
