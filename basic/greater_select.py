#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/3 21:36
# @Author  : Jasontang
# @Site    : 
# @File    : greater_select.py
# @ToDo    : greater与select函数

import tensorflow as tf

v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([4, 3, 2, 1])
# 使用交互式会话注册为默认会话
sess = tf.InteractiveSession()
# 比较v1和v2的大小
print(tf.greater(v1, v2).eval())
# tf.select变更为了tf.where
# tf.where有三个参数（元素级别上的操作）
# 第一个为选择条件根据
# 当条件为True时会选择第二个参数中的值，否则选第三个参数中的值
print(tf.where(tf.greater(v1, v2), v1, v2).eval())


