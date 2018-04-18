#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/3 19:19
# @Author  : Jasontang
# @Site    : 
# @File    : cross_entroy.py
# @ToDo    : 模拟交叉熵函数

import tensorflow as tf
import numpy as np


def log_(x):
    return np.log(x)


result = -tf.reduce_mean([[1.0, 0, 0]] * tf.log([[0.5, 0.4, 0.1]]))
with tf.Session() as sess:
    print(sess.run(result))
    print(-(1*log_(0.5) + 0*log_(0.4) + 0*log_(0.1)))      # 0.6931471805599453
    print(-(1*log_(0.5) + 0*log_(0.4) + 0*log_(0.1)) / 3)  # 0.23104906018664842
    # print(-(1*log_(0.8) + 0*log_(0.1) + 0*log_(0.1)))    # 0.2231435513142097
    print(result.eval())     # 0.23104906
    print(sess.run(result))  # 0.23104906
    print(tf.reduce_sum(tf.constant([[1, 2, 3]])).eval())    # 6
